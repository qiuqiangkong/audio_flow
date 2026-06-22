from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import librosa
import numpy as np
import soundfile
import torch
from torch import Tensor
from tqdm import tqdm

from audio_flow.solvers.euler import euler_solver
from audio_flow.utils import load_stereo, load_vae, parse_yaml
from train import get_model


TASK = "music source separation"


def resolve_path(path: str | Path, root: Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return root / path


def load_jsonl(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_latent(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as hf:
        return hf["latent"][:]


def iter_jsonl_inputs(jsonl_path: Path, root: Path):
    for meta in load_jsonl(jsonl_path):
        latent_path = resolve_path(meta["input"]["audio"]["latent_path"], root)
        duration = meta["input"]["audio"].get("duration")
        name = latent_path.with_suffix(".wav").name
        yield name, latent_path, duration


def iter_dataset_inputs(dataset_root: Path, input_filename: str):
    for input_path in sorted(dataset_root.glob(f"*/{input_filename}")):
        yield f"{input_path.parent.name}.wav", input_path, None


def chunk_starts(total: int, chunk: int, hop: int) -> list[int]:
    if total <= chunk:
        return [0]
    starts = list(range(0, max(total - chunk, 0) + 1, hop))
    last = total - chunk
    if starts[-1] != last:
        starts.append(last)
    return starts


def ola_weight(length: int, fade_in: int, fade_out: int) -> np.ndarray:
    weight = np.ones(length, dtype=np.float32)
    if fade_in > 0:
        phase = np.linspace(0.0, np.pi / 2.0, fade_in + 2, dtype=np.float32)[1:-1]
        weight[:fade_in] = np.sin(phase) ** 2
    if fade_out > 0:
        phase = np.linspace(np.pi / 2.0, 0.0, fade_out + 2, dtype=np.float32)[1:-1]
        weight[-fade_out:] = np.sin(phase) ** 2
    return weight


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if target_sr == orig_sr:
        return audio
    return librosa.resample(y=audio, orig_sr=orig_sr, target_sr=target_sr, axis=-1)


def match_rms(
    audio: np.ndarray,
    reference: np.ndarray,
    max_gain_db: float,
    peak_limit: float,
) -> np.ndarray:
    reference = librosa.util.fix_length(reference, size=audio.shape[-1], axis=-1)
    audio_rms = np.sqrt(np.mean(audio**2) + 1e-12)
    ref_rms = np.sqrt(np.mean(reference**2) + 1e-12)
    max_gain = 10 ** (max_gain_db / 20.0)
    gain = min(ref_rms / audio_rms, max_gain)
    audio = audio * gain
    peak = np.max(np.abs(audio))
    if peak > peak_limit > 0:
        audio = audio * (peak_limit / peak)
    return audio


def format_output_audio(audio: np.ndarray, mono_output: bool) -> np.ndarray:
    if not mono_output:
        return audio.T
    if audio.ndim == 1:
        return audio
    return np.mean(audio, axis=0)


@torch.inference_mode()
def infer_one_latent_chunk(
    *,
    model,
    vae,
    latent: np.ndarray,
    valid_frames: int,
    chunk_frames: int,
    solver_steps: int,
    device: str,
) -> np.ndarray:
    latent = librosa.util.fix_length(data=latent, size=chunk_frames, axis=0, constant_values=0.0)
    input_latent = Tensor(latent).to(device)[None, :, :]
    target_mask = torch.zeros((1, chunk_frames), dtype=torch.bool, device=device)
    target_mask[:, :valid_frames] = True
    data = {
        "task": [TASK],
        "input_latent": input_latent,
        "target_mask": target_mask,
    }

    noise = torch.randn(1, chunk_frames, vae.dim, device=device)
    controls = model.adapter(data)
    x_gen = euler_solver(model.base, noise, controls, n_steps=solver_steps)
    return vae.decode(x_gen).data.cpu().numpy()[0]


@torch.inference_mode()
def infer_latent_chunked(
    *,
    model,
    vae,
    latent: np.ndarray,
    chunk_duration: float,
    overlap_duration: float,
    solver_steps: int,
    device: str,
    target_duration: float | None = None,
) -> np.ndarray:
    chunk_frames = round(chunk_duration * vae.fps)
    overlap_frames = round(overlap_duration * vae.fps)
    if chunk_frames <= 0:
        raise ValueError("chunk_duration must be positive")
    if overlap_frames < 0 or overlap_frames >= chunk_frames:
        raise ValueError("overlap_duration must be >= 0 and < chunk_duration")

    hop_frames = chunk_frames - overlap_frames
    total_frames = latent.shape[0]
    target_samples = round((target_duration if target_duration is not None else total_frames / vae.fps) * vae.sr)
    audio_sum = np.zeros((2, target_samples), dtype=np.float32)
    weight_sum = np.zeros((target_samples,), dtype=np.float32)
    starts = chunk_starts(total_frames, chunk_frames, hop_frames)

    for i, start in enumerate(starts):
        latent_chunk = latent[start : start + chunk_frames]
        valid_frames = latent_chunk.shape[0]
        generated = infer_one_latent_chunk(
            model=model,
            vae=vae,
            latent=latent_chunk,
            valid_frames=valid_frames,
            chunk_frames=chunk_frames,
            solver_steps=solver_steps,
            device=device,
        )

        out_start = round(start / vae.fps * vae.sr)
        out_end = min(out_start + generated.shape[-1], target_samples)
        if out_end <= out_start:
            continue

        valid_samples = out_end - out_start
        fade_in = 0 if i == 0 else min(round(overlap_frames / vae.fps * vae.sr), valid_samples)
        fade_out = 0 if i == len(starts) - 1 else min(round(overlap_frames / vae.fps * vae.sr), valid_samples)
        weight = ola_weight(valid_samples, fade_in, fade_out)
        audio_sum[:, out_start:out_end] += generated[:, :valid_samples] * weight[None, :]
        weight_sum[out_start:out_end] += weight

    weight_sum = np.maximum(weight_sum, 1e-8)
    return audio_sum / weight_sum[None, :]


@torch.inference_mode()
def infer_audio_chunked(
    *,
    model,
    vae,
    audio_path: Path,
    chunk_duration: float,
    overlap_duration: float,
    solver_steps: int,
    device: str,
) -> np.ndarray:
    sr = vae.sr
    audio = load_stereo(str(audio_path), sr)
    target_samples = audio.shape[-1]
    chunk_samples = round(chunk_duration * sr)
    overlap_samples = round(overlap_duration * sr)
    if chunk_samples <= 0:
        raise ValueError("chunk_duration must be positive")
    if overlap_samples < 0 or overlap_samples >= chunk_samples:
        raise ValueError("overlap_duration must be >= 0 and < chunk_duration")

    hop_samples = chunk_samples - overlap_samples
    audio_sum = np.zeros((2, target_samples), dtype=np.float32)
    weight_sum = np.zeros((target_samples,), dtype=np.float32)
    starts = chunk_starts(target_samples, chunk_samples, hop_samples)

    for i, start in enumerate(starts):
        audio_chunk = audio[:, start : start + chunk_samples]
        valid_samples = audio_chunk.shape[-1]
        audio_chunk = librosa.util.fix_length(data=audio_chunk, size=chunk_samples, axis=-1)
        audio_chunk = Tensor(audio_chunk).to(device)
        latent = vae.encode(audio_chunk[None, :, :])[0].data.cpu().numpy()
        generated = infer_one_latent_chunk(
            model=model,
            vae=vae,
            latent=latent,
            valid_frames=latent.shape[0],
            chunk_frames=latent.shape[0],
            solver_steps=solver_steps,
            device=device,
        )

        out_start = start
        out_end = min(out_start + generated.shape[-1], target_samples)
        if out_end <= out_start:
            continue

        out_samples = out_end - out_start
        fade_in = 0 if i == 0 else min(overlap_samples, out_samples)
        fade_out = 0 if i == len(starts) - 1 else min(overlap_samples, out_samples)
        weight = ola_weight(out_samples, fade_in, fade_out)
        audio_sum[:, out_start:out_end] += generated[:, :out_samples] * weight[None, :]
        weight_sum[out_start:out_end] += weight

    weight_sum = np.maximum(weight_sum, 1e-8)
    return audio_sum / weight_sum[None, :]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./configs/mss/mss_musdb18hq.yaml")
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--jsonl_path", default="./jsonls/mss/test/musdb18hq.jsonl")
    parser.add_argument("--dataset_root")
    parser.add_argument("--input_filename", default="mixture.wav")
    parser.add_argument("--out_dir", default="./batch_results/mss_test_chunked")
    parser.add_argument("--chunk_duration", type=float, default=10.0)
    parser.add_argument("--overlap_duration", type=float, default=2.0)
    parser.add_argument("--solver_steps", type=int, default=100)
    parser.add_argument("--output_sr", type=int, default=None)
    parser.add_argument("--match_input_rms", action="store_true")
    parser.add_argument("--mono_output", action="store_true")
    parser.add_argument("--max_gain_db", type=float, default=20.0)
    parser.add_argument("--peak_limit", type=float, default=0.98)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--skip_existing", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    config_path = resolve_path(args.config, root)
    ckpt_path = resolve_path(args.ckpt_path, root)
    out_dir = resolve_path(args.out_dir, root)
    out_dir.mkdir(parents=True, exist_ok=True)

    configs = parse_yaml(str(config_path))
    device = configs["train"]["device"]

    model = get_model(configs, str(ckpt_path)).to(device)
    model.eval()
    vae = load_vae(configs["validate"].get("vae", "levo_vae")).to(device)
    vae.eval()
    output_sr = args.output_sr or vae.sr

    if args.dataset_root:
        inputs = list(iter_dataset_inputs(resolve_path(args.dataset_root, root), args.input_filename))
        mode = "audio"
    else:
        inputs = list(iter_jsonl_inputs(resolve_path(args.jsonl_path, root), root))
        mode = "latent"

    if args.limit:
        inputs = inputs[: args.limit]

    for name, in_path, duration in tqdm(inputs):
        out_path = out_dir / name
        if args.skip_existing and out_path.exists():
            continue

        if mode == "latent":
            audio = infer_latent_chunked(
                model=model,
                vae=vae,
                latent=load_latent(in_path),
                chunk_duration=args.chunk_duration,
                overlap_duration=args.overlap_duration,
                solver_steps=args.solver_steps,
                device=device,
                target_duration=duration,
            )
        else:
            audio = infer_audio_chunked(
                model=model,
                vae=vae,
                audio_path=in_path,
                chunk_duration=args.chunk_duration,
                overlap_duration=args.overlap_duration,
                solver_steps=args.solver_steps,
                device=device,
            )

        audio = resample_audio(audio, vae.sr, output_sr)
        if args.match_input_rms:
            if mode != "audio":
                raise ValueError("--match_input_rms requires --dataset_root audio inputs")
            reference = load_stereo(str(in_path), output_sr)
            audio = match_rms(audio, reference, args.max_gain_db, args.peak_limit)
        soundfile.write(file=out_path, data=format_output_audio(audio, args.mono_output), samplerate=output_sr)
        print(f"Write out to {out_path}")


if __name__ == "__main__":
    main()
