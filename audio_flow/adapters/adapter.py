import torch.nn as nn
from einops import rearrange
from torch import Tensor
import torch

from audio_flow.encoders.audio.clap import CLAPEncoder
# from audio_flow.encoders.text.char2 import CharEncoder2
from audio_flow.encoders.text.t5 import T5
# from audio_flow.models.aligner import Aligner
from audio_flow.utils import get_single_value
# from audio_flow.adapters.ttm import 


class TTMAdapter(nn.Module): 
    def __init__(self, dim: int, **kwargs):
        super().__init__()

        # T5
        self.t5 = T5()
        self.t5_fc = nn.Linear(self.t5.dim, dim)

        # CLAP
        self.clap = CLAPEncoder()
        self.clap_fc = nn.Linear(self.clap.dim, dim)

    def forward(self, data: dict) -> Tensor:

        # Task
        task, mask = self.t5(data["task"])  # (b, l, d)
        task = self.t5_fc(task)  # (b, l, d)
        task = mean_pool(task, mask, keepdims=True)  # (b, 1, d)
        
        # Clap
        clap = self.clap(data["prompt"])  # (b, d)
        clap = self.clap_fc(clap)[:, None, :]  # (b, 1, d)

        # Prompt
        prompt, prompt_mask = self.t5(data["prompt"])  # (b, l_v, d)
        prompt = self.t5_fc(prompt)  # (b, l1, d)

        # Build mask
        target_mask = data["target_mask"]  # (b, l_q)
        self_attn_mask = target_mask[:, None, None, :] * target_mask[:, None, :, None]  # (b, 1, l_q, l_q)
        cross_attn_mask = prompt_mask[:, None, None, :] * target_mask[:, None, :, None]  # (b, 1, l_q, l_v)
        assert check_masks([self_attn_mask, cross_attn_mask])
        
        c = task + clap
        seq = prompt

        controls = {
            "c": c,
            "seq": seq,
            "self_attn_mask": self_attn_mask,
            "cross_attn_mask": cross_attn_mask
        }

        return controls 


def mean_pool(x: Tensor, mask: Tensor, keepdims: bool) -> Tensor:
    r"""

    Args:
        x: (b, l, d)
        mask: (b, l)

    Returns:
        out: (b, d)
    """
    out = (x * mask[:, :, None]).sum(1) / mask.sum(dim=1, keepdims=True)
    
    if keepdims:
        out = out[:, None, :]

    return out



def check_masks(masks: list[Tensor]) -> bool:
    for mask in masks:
        if mask.dtype != torch.bool:
            return False
    return True



class Adapter(nn.Module): 
    def __init__(self, dim: int, **kwargs):
        super().__init__()

        # T5
        self.t5 = T5()
        self.t5_fc = nn.Linear(self.t5.dim, dim)

        # CLAP
        self.clap_text_encoder = CLAPTextEncoder()
        self.clap_fc = nn.Linear(self.clap_text_encoder.dim, dim)

    def forward(self, data: dict) -> Tensor:
        task = get_single_value(data["task"])


        if task == "text to music":
            return self.get_ttm_condition(data)

        else:
            raise ValueError(task)

    def get_ttm_condition(self, data: dict) -> dict:
        
        task = self.t5(data["task"])
        task = self.t5_fc(task).mean(dim=1, keepdims=True)  # (b, 1, d)

        prompt = self.t5(data["prompt"])  # (b, l', d)
        prompt = self.t5_fc(prompt)  # (b, l', d)

        # clap = self.clap_text(data["prompt"])  # (b, d)
        # clap = self.clap_fc(clap)[:, None, :]  # (b, 1, d)

        from IPython import embed; embed(using=False); os._exit(0) 

        if True:
            mask_latent = data["target_mask"]
            x_mask = mask_latent[:, None, None, :] * mask_latent[:, None, :, None]  # (b, 1, l_q, l_q)
            seq_mask = mask_text[:, None, None, :] * mask_latent[:, None, :, None]  # (b, 1, l_q, l_v)
            assert x_mask.dtype == torch.bool
            assert seq_mask.dtype == torch.bool
        else:  # tts_ljspeech_05b
            x_mask = None
            seq_mask = None

        controls = {
            "c": task + clap,
            "seq": prompt,
            "x_mask": x_mask,
            "seq_mask": seq_mask
        }
        # from IPython import embed; embed(using=False); os._exit(0)

        return controls 

        return task + prompt + clap

    # def get_ttm_condition(self, data: dict, length: int) -> Tensor:
    #     r"""Get text to music condition."""

    #     task = self.t5(data["task"])
    #     task = task.mean(dim=1, keepdims=True)  # (b, 1, d)

    #     prompt = self.t5(data["prompt"])  # (b, l', d)
    #     clap = self.clap_text(data["prompt"])  # (b, d)
    #     clap = self.clap_fc(clap)[:, None, :].repeat(1, length, 1)  # (b, l, d)

    #     return task + prompt + clap


'''
class TTM(nn.Module):
    def __init__(self, dim: int, max_length: int, **kwargs):
        super().__init__()

        self.t5 = T5()

    def forward(self, data: dict) -> Tensor:
        
        task = self.t5(data["task"])
        task = self.t5_fc(task).mean(dim=1, keepdims=True)  # (b, 1, d)

        prompt = self.t5(data["prompt"])  # (b, l', d)
        clap = self.clap_text(data["prompt"])  # (b, d)
        clap = self.clap_fc(clap)[:, None, :]  # (b, 1, d)

        from IPython import embed; embed(using=False); os._exit(0)

        if True:
            mask_latent = data["target_mask"]
            x_mask = mask_latent[:, None, None, :] * mask_latent[:, None, :, None]  # (b, 1, l_q, l_q)
            seq_mask = mask_text[:, None, None, :] * mask_latent[:, None, :, None]  # (b, 1, l_q, l_v)
            assert x_mask.dtype == torch.bool
            assert seq_mask.dtype == torch.bool
        else:  # tts_ljspeech_05b
            x_mask = None
            seq_mask = None

        controls = {
            "c": task + clap,
            "seq": prompt,
            "x_mask": x_mask,
            "seq_mask": seq_mask
        }
        # from IPython import embed; embed(using=False); os._exit(0)

        return controls 

        return task + prompt + clap
'''

'''
class Adapter(nn.Module): 
    def __init__(self, dim: int, max_length: int, **kwargs):
        super().__init__()

        # Text encoder
        self.t5_encoder = T5()
        self.t5_aligner = Aligner(self.t5_encoder.dim, dim, max_length)

        # TTS encoder
        self.char_encoder = CharEncoder2()
        self.char_embedder = nn.Embedding(self.char_encoder.vocab_size, dim)
        self.char_aligner = Aligner(dim, dim, max_length)
        
        # CLAP
        self.clap_text_encoder = CLAPTextEncoder()
        self.clap_fc = nn.Linear(self.clap_text_encoder.dim, dim)

        # VAE encoder
        self.audiovae_fc = nn.Linear(64, dim)

    def forward(self, data: dict) -> Tensor:
        r"""Get fixed length condition."""
        task = get_single_value(data["task"])
        
        if task == "text_to_music":
            return self.get_ttm_condition(data, length)
        
        elif task == "text_to_speech":
            return self.get_tts_condition(data)

        elif task == "text_to_audio":
            return self.get_tta_condition(data, length)

        elif task in ["music_source_separation", "mono_to_stereo", "super-resolution", "codec_to_music"]:
            return self.get_aligned_condition(data, length)

        else:
            raise ValueError(task)

    def get_ttm_condition(self, data: dict, length: int) -> Tensor:
        r"""Get text to music condition."""

        task = self.t5_encoder(data["task"])
        task = task.mean(dim=1, keepdims=True).repeat(1, length, 1)  # (b, l, d)

        prompt = self.t5_encoder(data["prompt"])  # (b, l', d)
        prompt = self.t5_aligner(prompt, length)  # (b, l, d)

        clap = self.clap_text_encoder(data["prompt"])  # (b, d)
        clap = self.clap_fc(clap)[:, None, :].repeat(1, length, 1)  # (b, l, d)

        return task + prompt + clap

    def get_tts_condition(self, data: dict) -> Tensor:
        r"""Get text to speech condition."""

        task = self.t5_encoder(data["task"])
        task = task.mean(dim=1)  # (b, d)

        prompt, mask_text = self.char_encoder(data["prompt"])  # (b, l_text, d), (b, l_text)
        prompt = self.char_embedder(prompt)  # (b, l_text, d)

        if True:
            mask_latent = data["target_mask"]
            x_mask = mask_latent[:, None, None, :] * mask_latent[:, None, :, None]  # (b, 1, l_q, l_q)
            seq_mask = mask_text[:, None, None, :] * mask_latent[:, None, :, None]  # (b, 1, l_q, l_v)
            assert x_mask.dtype == torch.bool
            assert seq_mask.dtype == torch.bool
        else:  # tts_ljspeech_05b
            x_mask = None
            seq_mask = None

        controls = {
            "c": task,
            "seq": prompt,
            "x_mask": x_mask,
            "seq_mask": seq_mask
        }
        # from IPython import embed; embed(using=False); os._exit(0)

        return controls 

    def get_tta_condition(self, data: dict, length: int) -> Tensor:
        r"""Get text to audio condition."""
        task = self.t5_encoder(data["task"])
        task = task.mean(dim=1, keepdims=True).repeat(1, length, 1)  # (b, l, d)

        prompt = self.t5_encoder(data["prompt"])  # (b, l', d)
        prompt = self.t5_aligner(prompt, length)  # (b, l, d)

        clap = self.clap_text_encoder(data["prompt"])  # (b, d)
        clap = self.clap_fc(clap)[:, None, :].repeat(1, length, 1)  # (b, l, d)

        return task + prompt + clap

    def get_aligned_condition(self, data: dict, length: int) -> Tensor:
        r"""Get aligned condition."""
        task = self.t5_encoder(data["task"])
        task = task.mean(dim=1, keepdims=True).repeat(1, length, 1)  # (b, l, d)

        instruction = self.t5_encoder(data["instruction"])
        instruction = instruction.mean(dim=1, keepdims=True).repeat(1, length, 1)  # (b, l, d)

        device = next(self.parameters()).device
        vae = rearrange(data["input_audio_latent"], 'b d t -> b t d').to(device)
        vae = self.audiovae_fc(vae)

        return task + instruction + vae


class Adapter3_ConvNeXt(nn.Module): 
    def __init__(self, dim: int, max_length: int, **kwargs):
        super().__init__()

        # Text encoder
        self.t5_encoder = T5()
        self.t5_aligner = Aligner(self.t5_encoder.dim, dim, max_length)

        # TTS encoder
        self.char_encoder = CharEncoder2()
        self.char_embedder = nn.Embedding(self.char_encoder.vocab_size, dim)
        self.char_aligner = ConvNeXt(dim)
        
        # CLAP
        self.clap_text_encoder = CLAPTextEncoder()
        self.clap_fc = nn.Linear(self.clap_text_encoder.dim, dim)

        # VAE encoder
        self.audiovae_fc = nn.Linear(64, dim)

    def forward(self, data: dict) -> Tensor:
        r"""Get fixed length condition."""
        task = get_single_value(data["task"])
        
        if task == "text_to_music":
            return self.get_ttm_condition(data)
        
        elif task == "text_to_speech":
            return self.get_tts_condition(data)

        elif task == "text_to_audio":
            return self.get_tta_condition(data, length)

        elif task in ["music_source_separation", "mono_to_stereo", "super-resolution", "codec_to_music"]:
            return self.get_aligned_condition(data, length)

        else:
            raise ValueError(task)

    def get_ttm_condition(self, data: dict) -> Tensor:
        r"""Get text to music condition."""

        task = self.t5_encoder(data["task"])
        task = task.mean(dim=1)  # (b, d)

        prompt = self.t5_encoder(data["prompt"])  # (b, l', d)

        clap = self.clap_text_encoder(data["prompt"])  # (b, d)
        clap = self.clap_fc(clap)

        if False:
            mask_latent = data["target_mask"]
            x_mask = mask_latent[:, None, None, :] * mask_latent[:, None, :, None]  # (b, 1, l_q, l_q)
            seq_mask = mask_text[:, None, None, :] * mask_latent[:, None, :, None]  # (b, 1, l_q, l_v)
            assert x_mask.dtype == torch.bool
            assert seq_mask.dtype == torch.bool
        else:  # tts_ljspeech_05b
            x_mask = None
            seq_mask = None

        controls = {
            "c": task + clap,
            "seq": prompt,
            "x_mask": x_mask,
            "seq_mask": seq_mask
        }

        return controls

    def get_tts_condition(self, data: dict) -> Tensor:
        r"""Get text to speech condition."""

        task = self.t5_encoder(data["task"])
        task = task.mean(dim=1)  # (b, d)

        prompt, mask_text = self.char_encoder(data["prompt"])  # (b, l_text, d), (b, l_text)
        prompt = self.char_embedder(prompt)  # (b, l_text, d)
        prompt = self.char_aligner(prompt)

        if True:
            mask_latent = data["target_mask"]
            x_mask = mask_latent[:, None, None, :] * mask_latent[:, None, :, None]  # (b, 1, l_q, l_q)
            seq_mask = mask_text[:, None, None, :] * mask_latent[:, None, :, None]  # (b, 1, l_q, l_v)
            assert x_mask.dtype == torch.bool
            assert seq_mask.dtype == torch.bool
        else:  # tts_ljspeech_05b
            x_mask = None
            seq_mask = None

        controls = {
            "c": task,
            "seq": prompt,
            "x_mask": x_mask,
            "seq_mask": seq_mask
        }
        # from IPython import embed; embed(using=False); os._exit(0)

        return controls 

    def get_tta_condition(self, data: dict, length: int) -> Tensor:
        r"""Get text to audio condition."""
        
        task = self.t5_encoder(data["task"])
        task = task.mean(dim=1)  # (b, d)

        prompt = self.t5_encoder(data["prompt"])  # (b, l', d)

        clap = self.clap_text_encoder(data["prompt"])  # (b, d)
        clap = self.clap_fc(clap)

        if False:
            mask_latent = data["target_mask"]
            x_mask = mask_latent[:, None, None, :] * mask_latent[:, None, :, None]  # (b, 1, l_q, l_q)
            seq_mask = mask_text[:, None, None, :] * mask_latent[:, None, :, None]  # (b, 1, l_q, l_v)
            assert x_mask.dtype == torch.bool
            assert seq_mask.dtype == torch.bool
        else:  # tts_ljspeech_05b
            x_mask = None
            seq_mask = None

        controls = {
            "c": task + clap,
            "seq": prompt,
            "x_mask": x_mask,
            "seq_mask": seq_mask
        }

        return controls

    def get_aligned_condition(self, data: dict, length: int) -> Tensor:
        r"""Get aligned condition."""
        task = self.t5_encoder(data["task"])
        task = task.mean(dim=1, keepdims=True).repeat(1, length, 1)  # (b, l, d)

        instruction = self.t5_encoder(data["instruction"])
        instruction = instruction.mean(dim=1, keepdims=True).repeat(1, length, 1)  # (b, l, d)

        device = next(self.parameters()).device
        vae = rearrange(data["input_audio_latent"], 'b d t -> b t d').to(device)
        vae = self.audiovae_fc(vae)

        return task + instruction + vae


class ConvNeXt(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.blocks = nn.ModuleList([ConvNeXtBlock(dim) for _ in range(3)])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)

        return x

class ConvNeXtBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

    def forward(self, x):
        # x: (b, l, d)

        residual = x
        x = rearrange(x, 'b l d -> b d l')
        x = self.dwconv(x)
        x = rearrange(x, 'b d l -> b l d')
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        return x + residual
'''