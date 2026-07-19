import torch.nn as nn
import numpy as np
import h5py
import torch

from .audio import normalize_dtype


def extract_and_save_video_features(
    video: np.ndarray, 
    chunk_frames: int, 
    model: nn.Module, 
    encoder_name: str,
    fps: float,
    out_path: str,
) -> None:
    
    out_path.parent.mkdir(parents=True, exist_ok=True)

    feat = extract_features_in_chunks(model, video, chunk_frames)  # (t, d)
    dtype = normalize_dtype(feat.dtype)

    with h5py.File(out_path, 'w') as hf:
        hf.create_dataset("data", data=feat, dtype=dtype)
        hf.attrs.create("fps", data=fps, dtype=float)
        hf.attrs.create("duration", data=feat.shape[0] / fps, dtype=float)
        hf.attrs.create("type", data=encoder_name)

        print(f"Write out to {out_path} {feat.shape}")


def extract_features_in_chunks(
    model: nn.Module, 
    images: np.array, 
    chunk_frames: int,
) -> np.array:
    r"""Convert audio into features.
    
    Args:
        model (nn.Module)
        x (np.ndarray): (t, c, h, w)

    Returns:
        out: (t, d)
    """
    device = next(model.parameters()).device
    outs = []
    i = 0
    
    while i < images.shape[0]:
        x = torch.from_numpy(images[i : i + chunk_frames, ...]).to(device)  # (t, c, h, w)

        with torch.no_grad():
            model.eval()
            out = model(x).cpu().numpy()  # (t, d)

        outs.append(out)
        i += chunk_frames

    return np.concatenate(outs, axis=0)
