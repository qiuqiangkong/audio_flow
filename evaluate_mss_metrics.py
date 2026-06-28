from __future__ import annotations

import argparse
import os
import csv
import math
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch
import soundfile as sf
from scipy import linalg, signal
from tqdm import tqdm


EPS = 1e-8


def to_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio.astype(np.float64)
    return np.mean(audio, axis=1).astype(np.float64)


def resample_audio(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return audio
    gcd = math.gcd(src_sr, dst_sr)
    up = dst_sr // gcd
    down = src_sr // gcd
    return signal.resample_poly(audio, up, down, axis=0)


def load_audio(path: Path, sr: int, mono: bool = True) -> np.ndarray:
    audio, file_sr = sf.read(path, always_2d=False)
    audio = resample_audio(audio, file_sr, sr)
    if mono:
        audio = to_mono(audio)
    return audio.astype(np.float64)


def align_pair(ref: np.ndarray, pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = min(len(ref), len(pred))
    return ref[:n], pred[:n]


def si_snr_db(ref: np.ndarray, pred: np.ndarray) -> float:
    ref, pred = align_pair(ref, pred)
    ref = ref - np.mean(ref)
    pred = pred - np.mean(pred)
    target = np.dot(pred, ref) * ref / (np.dot(ref, ref) + EPS)
    noise = pred - target
    return float(10.0 * np.log10((np.sum(target**2) + EPS) / (np.sum(noise**2) + EPS)))


def stft(audio: torch.Tensor, n_fft: int = 2048, hop_length: int = 512):
    hann_window = torch.hann_window(n_fft).to(audio.device)
    stft_spec = torch.stft(audio, n_fft, hop_length, window=hann_window, return_complex=True)
    stft_mag = torch.abs(stft_spec)
    stft_pha = torch.angle(stft_spec)

    return stft_mag, stft_pha


def lsd(ref: np.ndarray, pred: np.ndarray, sr: int, n_fft: int = 2048, hop: int = 512) -> float:
    ref, pred = align_pair(ref, pred)
    target = torch.from_numpy(ref).float()[None, :]
    pred = torch.from_numpy(pred).float()[None, :]

    sp = torch.log10(stft(pred, n_fft=n_fft, hop_length=hop)[0][:, :, :].square().clamp(1e-8))
    st = torch.log10(stft(target, n_fft=n_fft, hop_length=hop)[0][:, :, :].square().clamp(1e-8))
    return float((sp - st).square().mean(dim=1).sqrt().mean().item())


def pair_files(
    pred_dir: Path,
    dataset_root: Path,
    reference_filename: str,
) -> list[tuple[str, Path, Path]]:
    pairs = []
    fallback_names = []
    if reference_filename != "原始.wav":
        fallback_names.append("原始.wav")

    for pred_path in sorted(pred_dir.glob("*.wav")):
        song = pred_path.stem
        candidates = [dataset_root / song / reference_filename]
        candidates.extend(dataset_root / song / name for name in fallback_names)
        ref_path = next((path for path in candidates if path.exists()), None)
        if ref_path is not None:
            pairs.append((song, ref_path, pred_path))
        else:
            print(f"Skip missing reference: {song} -> {candidates[0]}")
    return pairs


def make_input_pairs(
    pairs: list[tuple[str, Path, Path]],
    dataset_root: Path,
    input_filename: str,
) -> list[tuple[str, Path, Path]]:
    input_pairs = []
    for song, ref_path, _ in pairs:
        input_path = dataset_root / song / input_filename
        if input_path.exists():
            input_pairs.append((song, ref_path, input_path))
        else:
            print(f"Skip missing input: {song} -> {input_path}")
    return input_pairs


def write_wav(path: Path, audio: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, sr)


def prepare_flat_dirs(
    pairs: list[tuple[str, Path, Path]],
    sr: int,
    work_dir: Path,
    name: str = "metric",
) -> tuple[Path, Path]:
    ref_dir = work_dir / name / "ref"
    pred_dir = work_dir / name / "pred"
    ref_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    for song, ref_path, pred_path in tqdm(pairs, desc="prepare metric wavs"):
        ref = load_audio(ref_path, sr=sr, mono=True)
        pred = load_audio(pred_path, sr=sr, mono=True)
        ref, pred = align_pair(ref, pred)
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", song).strip("_") + ".wav"
        write_wav(ref_dir / safe_name, ref, sr)
        write_wav(pred_dir / safe_name, pred, sr)

    return ref_dir, pred_dir


def compute_fad(ref_dir: Path, pred_dir: Path, project_root: Path) -> float | None:
    # The package top-level import also imports CLAP/transformers. Clear stale
    # localhost proxies and force local caches so VGGish FAD does not fail on an
    # unrelated Hugging Face metadata request.
    for key in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"]:
        os.environ.pop(key, None)
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    try:
        from frechet_audio_distance.fad import FrechetAudioDistance
    except Exception as e:
        raise RuntimeError(f"FAD package is installed but failed to import: {e}") from e

    ckpt_dir = project_root / "tools" / "fad_torchhub"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Reuse the VGGish torch hub cache already downloaded during installation,
    # but keep a project-local copy so later runs do not depend on ~/.cache.
    src_hub = Path.home() / ".cache" / "torch" / "hub"
    for name in ["harritaylor_torchvggish_master", "checkpoints"]:
        src = src_hub / name
        dst = ckpt_dir / name
        if src.exists() and not dst.exists():
            if src.is_dir():
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)

    fad = FrechetAudioDistance(
        ckpt_dir=str(ckpt_dir),
        model_name="vggish",
        sample_rate=16000,
        use_pca=False,
        use_activation=False,
        verbose=False,
    )
    return float(fad.score(str(ref_dir), str(pred_dir), dtype="float32"))


def parse_visqol_score(output: str) -> float | None:
    patterns = [
        r"MOS-LQO\s*:\s*([0-9.]+)",
        r"MOSLQO\s*:\s*([0-9.]+)",
        r"Similarity\s*:\s*([0-9.]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, output, flags=re.IGNORECASE)
        if match:
            return float(match.group(1))
    return None


def run_visqol_pair(
    visqol_bin: Path,
    ref_path: Path,
    pred_path: Path,
    speech_mode: bool,
    model_path: Path | None,
) -> float | None:
    cmd = [
        str(visqol_bin),
        "--reference_file",
        str(ref_path),
        "--degraded_file",
        str(pred_path),
        "--verbose",
    ]
    if speech_mode:
        cmd.append("--use_speech_mode")
    elif model_path is not None:
        cmd.extend(["--similarity_to_quality_model", str(model_path)])

    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    if proc.returncode != 0:
        print(f"ViSQOL failed for {pred_path.name}: {proc.stdout.strip()}")
        return None
    score = parse_visqol_score(proc.stdout)
    if score is None:
        print(f"ViSQOL score not found for {pred_path.name}: {proc.stdout.strip()}")
    return score


def compute_visqol(
    visqol_bin: Path | None,
    flat_ref_dir: Path,
    flat_pred_dir: Path,
    speech_mode: bool,
    model_path: Path | None,
) -> dict[str, float]:
    if visqol_bin is None:
        default_bin = Path(__file__).resolve().parent / "tools" / "visqol" / "bazel-bin" / "visqol"
        if default_bin.exists():
            visqol_bin = default_bin
        else:
            found = shutil.which("visqol")
            if found:
                visqol_bin = Path(found)
    if visqol_bin is None or not visqol_bin.exists():
        print("Skip ViSQOL: provide --visqol_bin or build tools/visqol/bazel-bin/visqol")
        return {}

    if model_path is None and not speech_mode:
        default_model = Path(__file__).resolve().parent / "tools" / "visqol" / "model" / "libsvm_nu_svr_model.txt"
        if default_model.exists():
            model_path = default_model
    if not speech_mode and (model_path is None or not model_path.exists()):
        print("Skip ViSQOL: provide --visqol_model pointing to libsvm_nu_svr_model.txt")
        return {}

    scores = {}
    for pred_path in tqdm(sorted(flat_pred_dir.glob("*.wav")), desc="ViSQOL"):
        ref_path = flat_ref_dir / pred_path.name
        score = run_visqol_pair(visqol_bin, ref_path, pred_path, speech_mode, model_path)
        if score is not None:
            scores[pred_path.stem] = score
    return scores


def summarize(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    return float(np.mean(arr)), float(np.std(arr))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_dir", default="batch_results/")
    parser.add_argument("--dataset_root", default="datasets/valid")
    parser.add_argument("--reference_filename", default="原始.wav")
    parser.add_argument("--input_filename", default="phone.wav")
    parser.add_argument("--compute_input_metrics", action="store_true")
    parser.add_argument("--out_csv", default="batch_results/valid_input.csv")
    parser.add_argument("--summary_path", default="batch_results/valid_input.txt")
    parser.add_argument("--metric_sr", type=int, default=16000)
    parser.add_argument("--visqol_sr", type=int, default=48000)
    parser.add_argument("--compute_fad", action="store_true")
    parser.add_argument("--compute_visqol", action="store_true")
    parser.add_argument("--visqol_bin", default="CCF_AATC_2026_1/tools/visqol/bazel-bin/visqol")
    parser.add_argument("--visqol_model", default="CCF_AATC_2026_1/tools/visqol/model/libsvm_nu_svr_model.txt")
    parser.add_argument("--visqol_speech_mode", action="store_true")
    parser.add_argument("--keep_metric_wavs")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    pred_dir = Path(args.pred_dir)
    if not pred_dir.is_absolute():
        pred_dir = root / pred_dir
    dataset_root = Path(args.dataset_root)
    out_csv = Path(args.out_csv)
    if not out_csv.is_absolute():
        out_csv = root / out_csv
    summary_path = Path(args.summary_path)
    if not summary_path.is_absolute():
        summary_path = root / summary_path

    pairs = pair_files(pred_dir, dataset_root, args.reference_filename)
    if not pairs:
        raise RuntimeError(f"No prediction/reference pairs found in {pred_dir}")

    input_pairs = make_input_pairs(pairs, dataset_root, args.input_filename) if args.compute_input_metrics else []
    input_by_song = {song: input_path for song, _, input_path in input_pairs}

    rows = []
    for song, ref_path, pred_path in tqdm(pairs, desc="SI-SNR/LSD"):
        ref = load_audio(ref_path, sr=args.metric_sr, mono=True)
        pred = load_audio(pred_path, sr=args.metric_sr, mono=True)
        ref, pred = align_pair(ref, pred)
        row = {
            "song": song,
            "ref_path": str(ref_path),
            "pred_path": str(pred_path),
            "seconds": len(ref) / args.metric_sr,
            "si_snr_db": si_snr_db(ref, pred),
            "lsd": lsd(ref, pred, sr=args.metric_sr),
        }
        input_path = input_by_song.get(song)
        if input_path is not None:
            ref_for_input = load_audio(ref_path, sr=args.metric_sr, mono=True)
            input_audio = load_audio(input_path, sr=args.metric_sr, mono=True)
            ref_for_input, input_audio = align_pair(ref_for_input, input_audio)
            row.update({
                "input_path": str(input_path),
                "input_seconds": len(ref_for_input) / args.metric_sr,
                "input_si_snr_db": si_snr_db(ref_for_input, input_audio),
                "input_lsd": lsd(ref_for_input, input_audio, sr=args.metric_sr),
            })
        rows.append(row)

    work_tmp = None
    metric_ref_dir = None
    metric_pred_dir = None
    visqol_ref_dir = None
    visqol_pred_dir = None
    if args.compute_fad or args.compute_visqol:
        if args.keep_metric_wavs:
            work_dir = Path(args.keep_metric_wavs)
            if not work_dir.is_absolute():
                work_dir = root / work_dir
            work_dir.mkdir(parents=True, exist_ok=True)
        else:
            work_tmp = tempfile.TemporaryDirectory()
            work_dir = Path(work_tmp.name)
        metric_input_ref_dir = None
        metric_input_pred_dir = None
        visqol_input_ref_dir = None
        visqol_input_pred_dir = None
        if args.compute_fad:
            metric_ref_dir, metric_pred_dir = prepare_flat_dirs(pairs, args.metric_sr, work_dir, name="fad")
            if input_pairs:
                metric_input_ref_dir, metric_input_pred_dir = prepare_flat_dirs(input_pairs, args.metric_sr, work_dir, name="fad_input")
        if args.compute_visqol:
            visqol_ref_dir, visqol_pred_dir = prepare_flat_dirs(pairs, args.visqol_sr, work_dir, name="visqol")
            if input_pairs:
                visqol_input_ref_dir, visqol_input_pred_dir = prepare_flat_dirs(input_pairs, args.visqol_sr, work_dir, name="visqol_input")

    fad_score = None
    input_fad_score = None
    if args.compute_fad and metric_ref_dir is not None and metric_pred_dir is not None:
        fad_score = compute_fad(metric_ref_dir, metric_pred_dir, root)
    if args.compute_fad and input_pairs and metric_input_ref_dir is not None and metric_input_pred_dir is not None:
        input_fad_score = compute_fad(metric_input_ref_dir, metric_input_pred_dir, root)

    visqol_scores = {}
    input_visqol_scores = {}
    if args.compute_visqol and visqol_ref_dir is not None and visqol_pred_dir is not None:
        visqol_scores = compute_visqol(
            Path(args.visqol_bin) if args.visqol_bin else None,
            visqol_ref_dir,
            visqol_pred_dir,
            speech_mode=args.visqol_speech_mode,
            model_path=Path(args.visqol_model) if args.visqol_model else None,
        )
        if input_pairs and visqol_input_ref_dir is not None and visqol_input_pred_dir is not None:
            input_visqol_scores = compute_visqol(
                Path(args.visqol_bin) if args.visqol_bin else None,
                visqol_input_ref_dir,
                visqol_input_pred_dir,
                speech_mode=args.visqol_speech_mode,
                model_path=Path(args.visqol_model) if args.visqol_model else None,
            )
        for row in rows:
            safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", row["song"]).strip("_")
            row["visqol"] = visqol_scores.get(safe_name)
            if input_visqol_scores:
                row["input_visqol"] = input_visqol_scores.get(safe_name)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "song", "seconds", "si_snr_db", "lsd", "visqol",
        "input_seconds", "input_si_snr_db", "input_lsd", "input_visqol",
        "ref_path", "pred_path", "input_path",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    si_mean, si_std = summarize([r["si_snr_db"] for r in rows])
    lsd_mean, lsd_std = summarize([r["lsd"] for r in rows])
    lines = [
        f"num_pairs: {len(rows)}",
        f"metric_sr: {args.metric_sr}",
        f"visqol_sr: {args.visqol_sr}",
        f"si_snr_db: mean={si_mean:.6f}, std={si_std:.6f}",
        f"lsd: mean={lsd_mean:.6f}, std={lsd_std:.6f}",
    ]
    input_rows = [r for r in rows if "input_si_snr_db" in r]
    if input_rows:
        input_si_mean, input_si_std = summarize([r["input_si_snr_db"] for r in input_rows])
        input_lsd_mean, input_lsd_std = summarize([r["input_lsd"] for r in input_rows])
        lines.extend([
            f"input_num_pairs: {len(input_rows)}",
            f"input_si_snr_db: mean={input_si_mean:.6f}, std={input_si_std:.6f}",
            f"input_lsd: mean={input_lsd_mean:.6f}, std={input_lsd_std:.6f}",
        ])
    if fad_score is not None:
        lines.append(f"fad: {fad_score:.6f}")
    if input_fad_score is not None:
        lines.append(f"input_fad: {input_fad_score:.6f}")
    if visqol_scores:
        visqol_mean, visqol_std = summarize(list(visqol_scores.values()))
        lines.append(f"visqol: mean={visqol_mean:.6f}, std={visqol_std:.6f}")
    if input_visqol_scores:
        input_visqol_mean, input_visqol_std = summarize(list(input_visqol_scores.values()))
        lines.append(f"input_visqol: mean={input_visqol_mean:.6f}, std={input_visqol_std:.6f}")

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"Write per-song metrics to {out_csv}")
    print(f"Write summary to {summary_path}")

    if work_tmp is not None:
        work_tmp.cleanup()


if __name__ == "__main__":
    main()
