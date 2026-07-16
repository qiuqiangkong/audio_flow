#!/usr/bin/env python3
"""Measure end-to-end inference parameters and MACs for CCF-AATC Track 1.

The standard baseline is split into the same modules that execute during
inference:

    waveform -> MMAudio VAE encoder -> condition adapter -> Euler sampling
    loop -> MMAudio VAE decoder/vocoder -> waveform

Each module is measured with ``ptflops.get_model_complexity_info`` using its
``aten`` backend. The Euler denoiser has the same tensor shapes at every step,
so it is measured once and its MAC value is multiplied by the number of actual
Euler model evaluations. With the repository's current ``euler_solver``,
``--solver-steps 100`` produces 99 denoiser evaluations.

For a participant's custom method, pass ``--model-factory module:function``.
The factory receives this script's argparse namespace and must return one
``torch.nn.Module`` whose ``forward(audio)`` executes the complete inference
pipeline, including all pretrained modules and every sampling iteration.
"""

from __future__ import annotations

import argparse
import importlib
import json
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

import torch
from torch import Tensor, nn
from ptflops import get_model_complexity_info

from audio_flow.solvers.euler import euler_solver
from audio_flow.utils import load_vae, parse_yaml
from train import get_model


@dataclass(frozen=True)
class ParameterStats:
    """Parameter counts for all components registered in an ``nn.Module``."""

    total: int
    trainable: int
    frozen: int
    by_top_level_component: dict[str, int]


def count_parameters(model: nn.Module) -> ParameterStats:
    """Count every registered parameter exactly once, including frozen ones."""

    # named_parameters() removes duplicated/shared parameters by default.
    parameters = list(model.named_parameters())
    total = sum(parameter.numel() for _, parameter in parameters)
    trainable = sum(
        parameter.numel() for _, parameter in parameters if parameter.requires_grad
    )

    by_component: dict[str, int] = {}
    seen: set[int] = set()
    for component_name, component in model.named_children():
        component_total = 0
        for parameter in component.parameters():
            parameter_id = id(parameter)
            if parameter_id not in seen:
                component_total += parameter.numel()
                seen.add(parameter_id)
        by_component[component_name] = component_total

    root_total = 0
    for _, parameter in model.named_parameters(recurse=False):
        parameter_id = id(parameter)
        if parameter_id not in seen:
            root_total += parameter.numel()
            seen.add(parameter_id)
    if root_total:
        by_component["<pipeline>"] = root_total

    return ParameterStats(
        total=total,
        trainable=trainable,
        frozen=total - trainable,
        by_top_level_component=by_component,
    )


def format_count(value: int | float, suffix: str = "") -> str:
    """Format a count using decimal units, for example ``1.234 GMACs``."""

    value = float(value)
    units = (("T", 1e12), ("G", 1e9), ("M", 1e6), ("K", 1e3))
    for prefix, scale in units:
        if abs(value) >= scale:
            return f"{value / scale:.3f} {prefix}{suffix}".strip()
    return f"{value:.0f} {suffix}".strip()


def _first_tensor(value: Any) -> Tensor | None:
    """Find the first tensor inside an ATen hook argument."""

    if isinstance(value, Tensor):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    return None


def scaled_dot_product_attention_macs(
    inputs: tuple[Any, ...], _: tuple[Any, ...]
) -> int:
    """Count QK^T and AV MACs for a fused attention ATen operation."""

    if len(inputs) < 3:
        return 0
    query = _first_tensor(inputs[0])
    key = _first_tensor(inputs[1])
    value = _first_tensor(inputs[2])
    if query is None or key is None or value is None:
        return 0
    if query.ndim < 2 or key.ndim < 2 or value.ndim < 2:
        return 0

    query_length = query.shape[-2]
    key_length = key.shape[-2]
    query_dim = query.shape[-1]
    value_dim = value.shape[-1]
    batch_heads = query.numel() // (query_length * query_dim)
    return int(batch_heads * query_length * key_length * (query_dim + value_dim))


def convolution_1d_macs(
    inputs: tuple[Any, ...], outputs: tuple[Any, ...], *, transposed: bool
) -> int:
    """Count Conv1d/ConvTranspose1d MACs exposed as high-level ATen ops.

    PTFLops already handles ``aten.convolution``, but calls made inside
    ``torch.inference_mode`` can remain as ``aten.conv1d`` or
    ``aten.conv_transpose1d``. MMAudio's VAE and BigVGAN use those paths, so
    omitting these rules silently drops nearly all of their convolution MACs.
    """

    if len(inputs) < 2:
        return 0
    input_tensor = _first_tensor(inputs[0])
    weight = _first_tensor(inputs[1])
    output = _first_tensor(outputs)
    bias = _first_tensor(inputs[2]) if len(inputs) > 2 else None
    if input_tensor is None or weight is None or output is None:
        return 0

    spatial_shape = input_tensor.shape[2:] if transposed else output.shape[2:]
    spatial_positions = 1
    for size in spatial_shape:
        spatial_positions *= int(size)

    macs = int(input_tensor.shape[0]) * weight.numel() * spatial_positions
    if bias is not None:
        macs += output.numel()
    return int(macs)


def conv1d_macs(inputs: tuple[Any, ...], outputs: tuple[Any, ...]) -> int:
    return convolution_1d_macs(inputs, outputs, transposed=False)


def conv_transpose1d_macs(
    inputs: tuple[Any, ...], outputs: tuple[Any, ...]
) -> int:
    return convolution_1d_macs(inputs, outputs, transposed=True)


def ptflops_custom_aten_hooks() -> dict[Any, Any]:
    """Return PTFLops rules missing from its default ATen operator table."""

    hooks: dict[Any, Any] = {}
    operation_names = (
        "scaled_dot_product_attention",
        "_scaled_dot_product_attention_math",
        "_scaled_dot_product_flash_attention",
        "_scaled_dot_product_flash_attention_for_cpu",
        "_scaled_dot_product_efficient_attention",
        "_scaled_dot_product_cudnn_attention",
    )
    for operation_name in operation_names:
        operation = getattr(torch.ops.aten, operation_name, None)
        if operation is not None:
            hooks[operation] = scaled_dot_product_attention_macs

    convolution_hooks = {
        "conv1d": conv1d_macs,
        "conv_transpose1d": conv_transpose1d_macs,
    }
    for operation_name, hook in convolution_hooks.items():
        operation = getattr(torch.ops.aten, operation_name, None)
        if operation is not None:
            hooks[operation] = hook
    return hooks


class AudioFlowMSSInference(nn.Module):
    """The complete single-chunk AudioFlow music-restoration inference graph."""

    def __init__(
        self,
        flow_model: nn.Module,
        vae: nn.Module,
        task: str,
        solver_steps: int,
    ) -> None:
        super().__init__()
        if solver_steps < 1:
            raise ValueError("solver_steps must be at least 1")
        self.flow_model = flow_model
        self.vae = vae
        self.task = task
        self.solver_steps = solver_steps

    @property
    def denoiser_forward_calls(self) -> int:
        """Number of base-model calls made by the current Euler solver."""

        # ``euler_solver`` uses len(torch.linspace(..., n_steps)) - 1 updates.
        return max(self.solver_steps - 1, 0)

    def forward(self, audio: Tensor) -> Tensor:
        """Restore one waveform chunk and return its decoded audio."""

        input_latent = self.vae.encode(audio)
        batch_size, frames, _ = input_latent.shape
        target_mask = torch.ones(
            (batch_size, frames), dtype=torch.bool, device=input_latent.device
        )
        data = {
            "task": [self.task] * batch_size,
            "input_latent": input_latent,
            "target_mask": target_mask,
        }

        # The particular noise values do not affect MACs, while a fixed zero
        # tensor keeps complexity evaluation deterministic and reproducible.
        noise = torch.zeros(
            (batch_size, frames, self.vae.dim),
            dtype=input_latent.dtype,
            device=input_latent.device,
        )
        controls = self.flow_model.adapter(data)
        generated_latent = euler_solver(
            self.flow_model.base,
            noise,
            controls,
            n_steps=self.solver_steps,
        )
        return self.vae.decode(generated_latent)


class BaselineVAEEncoder(nn.Module):
    """The waveform-to-latent part of the MMAudio VAE."""

    def __init__(self, vae: nn.Module) -> None:
        super().__init__()
        self.vae = vae

    def forward(self, audio: Tensor) -> Tensor:
        return self.vae.encode(audio)


class BaselineConditionAdapter(nn.Module):
    """The MSS adapter, including its pretrained T5 encoder."""

    def __init__(self, adapter: nn.Module, task: str) -> None:
        super().__init__()
        self.adapter = adapter
        self.task = task

    def forward(self, input_latent: Tensor) -> dict[str, Tensor]:
        batch_size, frames, _ = input_latent.shape
        target_mask = torch.ones(
            (batch_size, frames), dtype=torch.bool, device=input_latent.device
        )
        data = {
            "task": [self.task] * batch_size,
            "input_latent": input_latent,
            "target_mask": target_mask,
        }
        return self.adapter(data)


class BaselineDenoiser(nn.Module):
    """One Euler denoiser evaluation with shape-equivalent controls."""

    def __init__(self, base: nn.Module, hidden_dim: int) -> None:
        super().__init__()
        self.base = base
        self.hidden_dim = hidden_dim

    def forward(self, noise: Tensor) -> Tensor:
        batch_size, frames, _ = noise.shape
        controls = {
            "c": torch.zeros(
                (batch_size, 1, self.hidden_dim),
                dtype=noise.dtype,
                device=noise.device,
            ),
            "seq": torch.zeros(
                (batch_size, frames, self.hidden_dim),
                dtype=noise.dtype,
                device=noise.device,
            ),
            "self_attn_mask": torch.ones(
                (batch_size, 1, frames, frames),
                dtype=torch.bool,
                device=noise.device,
            ),
            "cross_attn_mask": torch.eye(
                frames, dtype=torch.bool, device=noise.device
            )[None, None, :, :],
        }
        time = torch.zeros((), dtype=noise.dtype, device=noise.device)
        return self.base(time, noise, controls)


class BaselineVAEDecoderVocoder(nn.Module):
    """The latent-to-waveform MMAudio VAE decoder and BigVGAN vocoder."""

    def __init__(self, vae: nn.Module) -> None:
        super().__init__()
        self.vae = vae

    def forward(self, latent: Tensor) -> Tensor:
        return self.vae.decode(latent)


class ParameterContainer(nn.Module):
    """Register one component so PTFLops can report its parameter count.

    The component intentionally is not called in ``forward``. PTFLops obtains
    the parameter count from registered submodules, while this wrapper keeps
    the corresponding MAC count at zero and avoids double-counting a shared
    VAE in both encoder and decoder measurements.
    """

    def __init__(self, component: nn.Module) -> None:
        super().__init__()
        self.component = component

    def forward(self, value: Tensor) -> Tensor:
        return value


def parse_factory(specification: str) -> Callable[[argparse.Namespace], nn.Module]:
    """Import a participant factory specified as ``python.module:function``."""

    if ":" not in specification:
        raise ValueError(
            "--model-factory must have the form 'python.module:function'"
        )
    module_name, function_name = specification.split(":", maxsplit=1)
    module = importlib.import_module(module_name)
    factory = getattr(module, function_name, None)
    if factory is None or not callable(factory):
        raise ValueError(
            f"Could not find callable {function_name!r} in module {module_name!r}"
        )
    return factory


def load_checkpoint(model: nn.Module, checkpoint_path: str | None) -> None:
    """Load a checkpoint on CPU; weights do not change model complexity."""

    if not checkpoint_path:
        return
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    incompatible = model.load_state_dict(checkpoint, strict=False)
    print(f"Loaded checkpoint: {checkpoint_path}")
    if incompatible.missing_keys:
        print(f"  Missing keys: {len(incompatible.missing_keys)}")
    if incompatible.unexpected_keys:
        print(f"  Unexpected keys: {len(incompatible.unexpected_keys)}")


def build_baseline_pipeline(args: argparse.Namespace) -> tuple[nn.Module, int, float]:
    """Build the repository baseline and resolve its standard input settings."""

    configs = parse_yaml(args.config)
    flow_model = get_model(configs, ckpt_path=None)
    load_checkpoint(flow_model, args.ckpt_path)

    vae_name = args.vae or configs.get("validate", {}).get("vae", "levo_vae")
    vae = load_vae(vae_name)
    pipeline = AudioFlowMSSInference(
        flow_model=flow_model,
        vae=vae,
        task=args.task,
        solver_steps=args.solver_steps,
    )

    duration = args.duration
    if duration is None:
        duration = float(configs.get("clip_duration", 10.0))
    sample_rate = args.sample_rate or int(vae.sr)
    return pipeline, sample_rate, duration


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable")
    return device


@contextmanager
def include_frozen_parameters_in_ptflops(model: nn.Module) -> Iterator[None]:
    """Make PTFLops include every registered parameter in its ``params`` value.

    PTFLops normally skips ``requires_grad=False`` parameters. For a deployment
    complexity rule, frozen T5/VAE/vocoder weights still have to be counted.
    This context only changes the flags while the no-gradient complexity pass
    runs and restores every original value afterwards.
    """

    original_flags = [
        (parameter, parameter.requires_grad) for parameter in model.parameters()
    ]
    try:
        for parameter, _ in original_flags:
            parameter.requires_grad_(True)
        yield
    finally:
        for parameter, original_flag in original_flags:
            parameter.requires_grad_(original_flag)


def profile_with_ptflops(
    model: nn.Module,
    input_shape: tuple[int, ...],
    input_name: str,
    device: torch.device,
    verbose: bool,
    custom_aten_hooks: dict | None = None,
) -> tuple[int, int]:
    """Return ``(MACs, parameters)`` from ``ptflops.get_model_complexity_info``.

    ``input_shape`` excludes the batch dimension, exactly as required by
    PTFLops. ``input_name`` is the keyword accepted by the module's
    ``forward`` method, for example ``audio``, ``input_latent`` or ``noise``.
    """

    first_parameter = next(model.parameters(), None)
    dtype = (
        first_parameter.dtype
        if first_parameter is not None and first_parameter.is_floating_point()
        else torch.float32
    )

    def input_constructor(resolution: tuple[int, ...]) -> dict[str, Tensor]:
        return {
            input_name: torch.zeros(
                (1, *resolution), dtype=dtype, device=device
            )
        }

    # ``aten`` is selected deliberately: unlike the legacy module-hook backend,
    # it also covers functional matrix operations used by Transformer models.
    aten_hooks = ptflops_custom_aten_hooks()
    if custom_aten_hooks:
        aten_hooks.update(custom_aten_hooks)

    with include_frozen_parameters_in_ptflops(model):
        macs, params = get_model_complexity_info(
            model=model,
            input_res=input_shape,
            input_constructor=input_constructor,
            print_per_layer_stat=False,
            as_strings=False,
            verbose=verbose,
            backend="aten",
            custom_modules_hooks=aten_hooks,
        )
    if macs is None or params is None:
        raise RuntimeError("ptflops could not finish the complexity analysis")
    return int(macs), int(params)


def profile_parameter_group(component: nn.Module, device: torch.device) -> int:
    """Get one non-overlapping component's parameter count through PTFLops."""

    _, params = profile_with_ptflops(
        ParameterContainer(component),
        input_shape=(1,),
        input_name="value",
        device=device,
        verbose=False,
    )
    return params


def infer_mmaudio_latent_frames(
    vae: nn.Module, sample_count: int, sample_rate: int
) -> int:
    """Infer the MMAudio latent length without executing a VAE forward pass."""

    fps = getattr(vae, "fps", None)
    if fps is None:
        raise ValueError(
            "Component-wise baseline profiling requires a VAE with an `fps` "
            "attribute. Use --baseline-profile-mode whole for this VAE."
        )
    frames = int(sample_count * float(fps) / sample_rate)
    if frames < 1:
        raise ValueError("The selected audio duration produces zero latent frames")
    return frames


def profile_baseline_components(
    pipeline: AudioFlowMSSInference,
    input_shape: tuple[int, int, int],
    sample_rate: int,
    device: torch.device,
    verbose: bool,
) -> tuple[int, list[dict[str, Any]], dict[str, int]]:
    """Profile each baseline module with PTFLops and sum the executed MACs."""

    _, channels, sample_count = input_shape
    latent_dim = getattr(pipeline.vae, "dim", None)
    if latent_dim is None:
        raise ValueError(
            "Component-wise baseline profiling requires a VAE with a `dim` "
            "attribute. Use --baseline-profile-mode whole for this VAE."
        )
    latent_frames = infer_mmaudio_latent_frames(
        pipeline.vae, sample_count, sample_rate
    )

    base = pipeline.flow_model.base
    fc_in = getattr(base, "fc_in", None)
    hidden_dim = getattr(fc_in, "out_features", None)
    if hidden_dim is None:
        raise ValueError(
            "The default component-wise profiler expects the baseline "
            "Transformer's `fc_in` layer. Use --baseline-profile-mode whole "
            "for a different base model."
        )

    component_specs = (
        (
            "MMAudio VAE encoder",
            BaselineVAEEncoder(pipeline.vae),
            (channels, sample_count),
            "audio",
            1,
        ),
        (
            "Condition adapter (including T5)",
            BaselineConditionAdapter(pipeline.flow_model.adapter, pipeline.task),
            (latent_frames, int(latent_dim)),
            "input_latent",
            1,
        ),
        (
            "Flow denoiser",
            BaselineDenoiser(base, int(hidden_dim)),
            (latent_frames, int(latent_dim)),
            "noise",
            pipeline.denoiser_forward_calls,
        ),
        (
            "MMAudio VAE decoder and BigVGAN vocoder",
            BaselineVAEDecoderVocoder(pipeline.vae),
            (latent_frames, int(latent_dim)),
            "latent",
            1,
        ),
    )

    component_macs: list[dict[str, Any]] = []
    for name, module, module_input_shape, input_name, calls in component_specs:
        one_call_macs, _ = profile_with_ptflops(
            module,
            input_shape=module_input_shape,
            input_name=input_name,
            device=device,
            verbose=verbose,
        )
        component_macs.append(
            {
                "name": name,
                "macs_per_call": one_call_macs,
                "calls": calls,
                "macs_total": one_call_macs * calls,
            }
        )

    component_parameters = {
        "Flow denoiser": profile_parameter_group(base, device),
        "Condition adapter (including T5)": profile_parameter_group(
            pipeline.flow_model.adapter, device
        ),
        "MMAudio VAE, decoder and BigVGAN": profile_parameter_group(
            pipeline.vae, device
        ),
    }
    total_macs = sum(component["macs_total"] for component in component_macs)
    return total_macs, component_macs, component_parameters


def get_submission_custom_aten_hooks(pipeline: nn.Module) -> dict:
    """Read optional custom ATen MAC rules from a participant pipeline.

    A submission can define ``ptflops_custom_aten_hooks(self)`` and return the
    ``{torch.ops.aten.<operation>: hook}`` mapping expected by ptflops.  This is
    useful for a custom fused PyTorch operator that ptflops otherwise treats as
    a zero-op.
    """

    provider = getattr(pipeline, "ptflops_custom_aten_hooks", None)
    if provider is None:
        return {}
    if not callable(provider):
        raise TypeError("pipeline.ptflops_custom_aten_hooks must be callable")
    hooks = provider()
    if not isinstance(hooks, dict):
        raise TypeError("pipeline.ptflops_custom_aten_hooks must return a dict")
    return hooks


def make_report(
    *,
    pipeline: nn.Module,
    parameters,
    macs: int,
    input_shape: tuple[int, int, int],
    sample_rate: int,
    duration: float,
    device: torch.device,
    baseline: bool,
    component_macs: list[dict[str, Any]] | None = None,
    component_parameters: dict[str, int] | None = None,
    profile_mode: str | None = None,
) -> dict:
    report = {
        "metric": {
            "parameters": "all registered parameters, including pretrained/frozen modules",
            "macs": (
                "ptflops ATen-backend MACs for one complete inference call; "
                "fused scaled-dot-product attention and high-level Conv1d operations "
                "are covered by custom ptflops rules"
            ),
            "flops": "2 * MACs (one MAC is one multiply-accumulate pair)",
            "activations": (
                "not reported: ptflops does not expose an activation-count metric"
            ),
        },
        "input": {
            "shape": list(input_shape),
            "sample_rate_hz": sample_rate,
            "duration_seconds": duration,
            "device": str(device),
        },
        "parameters": {
            "total": parameters.total,
            "trainable": parameters.trainable,
            "frozen": parameters.frozen,
            "by_top_level_component": parameters.by_top_level_component,
        },
        "macs": {
            "total": macs,
            "approx_flops": 2 * macs,
            "counter": "ptflops (backend=aten)",
        },
    }
    if component_macs is not None or component_parameters is not None:
        report["components"] = {
            "parameters": component_parameters or {},
            "macs": component_macs or [],
        }
    if baseline:
        component_note = (
            "Each unique baseline component was measured with PTFLops. The "
            "single denoiser measurement was multiplied by the number of "
            "Euler model evaluations."
            if profile_mode == "components"
            else "The complete inference loop was measured in one PTFLops call."
        )
        report["sampling"] = {
            "solver_grid_points_argument": pipeline.solver_steps,
            "actual_denoiser_forward_calls": pipeline.denoiser_forward_calls,
            "profile_mode": profile_mode,
            "note": component_note,
        }
    return report


def print_report(report: dict) -> None:
    input_info = report["input"]
    parameters = report["parameters"]
    macs = report["macs"]

    print("=" * 72)
    print("End-to-end inference complexity")
    print(
        "Input: "
        f"shape={tuple(input_info['shape'])}, "
        f"{input_info['duration_seconds']:.3f}s at "
        f"{input_info['sample_rate_hz']} Hz, "
        f"device={input_info['device']}"
    )
    print("-" * 72)
    print(
        "Parameters (all components): "
        f"{parameters['total']:,} ({format_count(parameters['total'])})"
    )
    print(f"  trainable: {format_count(parameters['trainable'])}")
    print(f"  frozen:    {format_count(parameters['frozen'])}")
    for name, value in parameters["by_top_level_component"].items():
        print(f"  {name}: {format_count(value)}")
    components = report.get("components")
    if components and components["parameters"]:
        print("  PTFLops component parameters:")
        for name, value in components["parameters"].items():
            print(f"    {name}: {format_count(value)}")
    print(
        "MACs (whole inference):    "
        f"{macs['total']:,} ({format_count(macs['total'], 'MACs')})"
    )
    print(
        "FLOPs (2 x MACs):          "
        f"{macs['approx_flops']:,} "
        f"({format_count(macs['approx_flops'], 'FLOPs')})"
    )

    if components and components["macs"]:
        print("MACs by executed component:")
        for component in components["macs"]:
            name = component["name"]
            per_call = format_count(component["macs_per_call"], "MACs")
            calls = component["calls"]
            total = format_count(component["macs_total"], "MACs")
            if calls == 1:
                print(f"  {name}: {total}")
            else:
                print(f"  {name}: {per_call} x {calls} = {total}")

    if "sampling" in report:
        sampling = report["sampling"]
        print(
            "Sampling: "
            f"--solver-steps={sampling['solver_grid_points_argument']} -> "
            f"{sampling['actual_denoiser_forward_calls']} denoiser forward calls"
        )
        print(f"Profile mode: {sampling['profile_mode']}")

    print(
        "Counter: ptflops (ATen backend; custom fused-SDPA and 1D-convolution rules)"
    )
    print("Activations: not reported by ptflops")
    print("=" * 72)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure all registered parameters and executed end-to-end MACs."
    )
    parser.add_argument(
        "--config",
        default="./configs/mss/mss_musdb18hq.yaml",
        help="AudioFlow configuration used by the default baseline pipeline.",
    )
    parser.add_argument(
        "--ckpt-path",
        default=None,
        help="Optional flow-model checkpoint. It does not affect the architecture.",
    )
    parser.add_argument(
        "--vae",
        default=None,
        help="Override the VAE selected in the config (default: config's validation VAE).",
    )
    parser.add_argument("--task", default="music source separation")
    parser.add_argument(
        "--solver-steps",
        type=int,
        default=100,
        help="Value passed to the current Euler solver (100 means 99 model calls).",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Input duration in seconds (default: config clip_duration, normally 10).",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=None,
        help="Input sample rate in Hz (default: selected VAE sample rate).",
    )
    parser.add_argument("--channels", type=int, default=2)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Use GPU when available by default; CPU is supported but slower.",
    )
    parser.add_argument(
        "--baseline-profile-mode",
        choices=("components", "whole"),
        default="components",
        help=(
            "For the built-in baseline: profile each module and multiply the "
            "single denoiser result by its Euler call count (default), or run "
            "the whole loop in one PTFLops call (slow cross-check)."
        ),
    )
    parser.add_argument(
        "--model-factory",
        default=None,
        metavar="MODULE:FUNCTION",
        help=(
            "Participant hook: factory(args) -> nn.Module with forward(audio) that "
            "contains the complete inference pipeline."
        ),
    )
    parser.add_argument(
        "--json-out",
        default=None,
        help="Optional path for a machine-readable report.",
    )
    parser.add_argument(
        "--verbose-ptflops",
        action="store_true",
        help="Print PTFLops operations treated as zero-op; useful for custom models.",
    )
    args = parser.parse_args()

    device = resolve_device(args.device)
    if args.model_factory:
        factory = parse_factory(args.model_factory)
        pipeline = factory(args)
        if not isinstance(pipeline, nn.Module):
            raise TypeError("The custom model factory must return torch.nn.Module")
        duration = 10.0 if args.duration is None else args.duration
        sample_rate = 16000 if args.sample_rate is None else args.sample_rate
        baseline = False
    else:
        pipeline, sample_rate, duration = build_baseline_pipeline(args)
        baseline = True

    if duration <= 0:
        raise ValueError("--duration must be positive")
    if args.channels <= 0:
        raise ValueError("--channels must be positive")

    pipeline = pipeline.to(device).eval()
    sample_count = round(duration * sample_rate)
    input_shape = (1, args.channels, sample_count)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    parameters = count_parameters(pipeline)
    component_macs: list[dict[str, Any]] | None = None
    component_parameters: dict[str, int] | None = None
    profile_mode: str | None = None

    if baseline and args.baseline_profile_mode == "components":
        profile_mode = "components"
        macs, component_macs, component_parameters = profile_baseline_components(
            pipeline,
            input_shape=input_shape,
            sample_rate=sample_rate,
            device=device,
            verbose=args.verbose_ptflops,
        )
        ptflops_parameter_total = sum(component_parameters.values())
    else:
        profile_mode = "whole" if baseline else None
        custom_aten_hooks = get_submission_custom_aten_hooks(pipeline)
        macs, ptflops_parameter_total = profile_with_ptflops(
            pipeline,
            input_shape=(args.channels, sample_count),
            input_name="audio",
            device=device,
            verbose=args.verbose_ptflops,
            custom_aten_hooks=custom_aten_hooks,
        )

    if ptflops_parameter_total != parameters.total:
        raise RuntimeError(
            "PTFLops parameter total does not match the registered-parameter "
            f"total ({ptflops_parameter_total} != {parameters.total})."
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    report = make_report(
        pipeline=pipeline,
        parameters=parameters,
        macs=macs,
        input_shape=input_shape,
        sample_rate=sample_rate,
        duration=duration,
        device=device,
        baseline=baseline,
        component_macs=component_macs,
        component_parameters=component_parameters,
        profile_mode=profile_mode,
    )
    print_report(report)

    if args.json_out:
        json_path = Path(args.json_out)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with json_path.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        print(f"Wrote JSON report: {json_path}")


if __name__ == "__main__":
    main()
