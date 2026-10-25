"""Pluggable multimodal encoders that project into the dense model's token stream.

Each encoder maps raw inputs to [B, N, d_model] "prefix tokens"; ArtemisDense.forward_multimodal
concatenates them before the text embeddings, so the same causal dense trunk sees image/audio
tokens first. Attach with model.attach_encoder("image", ImagePatchEncoder(...)).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ImagePatchEncoder(nn.Module):
    """ViT-style patch embedding: Conv2d(kernel=stride=patch) -> + learned pos -> Linear."""

    def __init__(self, d_model: int, img_size: int = 32, patch: int = 8, in_ch: int = 3):
        super().__init__()
        if img_size % patch:
            raise ValueError("img_size must be divisible by patch")
        self.d_model, self.n_patches = d_model, (img_size // patch) ** 2
        self.proj = nn.Conv2d(in_ch, d_model, kernel_size=patch, stride=patch)
        self.pos = nn.Parameter(torch.zeros(1, self.n_patches, d_model))
        self.out = nn.Linear(d_model, d_model)

    def forward(self, img):                       # [B, C, H, W]
        x = self.proj(img).flatten(2).transpose(1, 2)    # [B, N, d]
        return self.out(x + self.pos)


def mel_filterbank(n_fft: int, n_mels: int, sample_rate: int) -> torch.Tensor:
    """Triangular HTK-mel filters [n_mels, n_fft//2+1] (no torchaudio dependency)."""
    hz2mel = lambda f: 2595 * math.log10(1 + f / 700)
    mel2hz = lambda m: 700 * (10 ** (m / 2595) - 1)
    mels = torch.linspace(hz2mel(0), hz2mel(sample_rate / 2), n_mels + 2)
    hz = torch.tensor([mel2hz(float(m)) for m in mels])
    bins = torch.linspace(0, sample_rate / 2, n_fft // 2 + 1)
    fb = torch.zeros(n_mels, bins.numel())
    for i in range(n_mels):
        l, c, r = hz[i], hz[i + 1], hz[i + 2]
        fb[i] = torch.clamp(torch.minimum((bins - l) / (c - l), (r - bins) / (r - c)), min=0)
    return fb


def log_mel(wave: torch.Tensor, sample_rate: int = 16000, n_fft: int = 400, hop: int = 160,
            n_mels: int = 80) -> torch.Tensor:
    """[B, samples] -> [B, n_mels, frames] log-mel spectrogram."""
    spec = torch.stft(wave, n_fft, hop, window=torch.hann_window(n_fft, device=wave.device),
                      return_complex=True).abs() ** 2
    mel = mel_filterbank(n_fft, n_mels, sample_rate).to(wave.device) @ spec
    return torch.log(mel.clamp_min(1e-10))


class AudioLogMelEncoder(nn.Module):
    """Whisper-style conv frontend: two Conv1d (second stride 2) + GELU over log-mel frames -> Linear."""

    def __init__(self, d_model: int, n_mels: int = 80):
        super().__init__()
        self.d_model = d_model
        self.conv1 = nn.Conv1d(n_mels, d_model, 3, padding=1)
        self.conv2 = nn.Conv1d(d_model, d_model, 3, stride=2, padding=1)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, mel):                       # [B, n_mels, frames]
        x = F.gelu(self.conv2(F.gelu(self.conv1(mel))))
        return self.out(x.transpose(1, 2))        # [B, ceil(frames/2), d]
