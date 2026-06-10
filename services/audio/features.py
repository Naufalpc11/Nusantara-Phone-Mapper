"""Audio decoding and log-Mel feature extraction."""

import math
from functools import lru_cache

import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio.functional as audio_functional


def load_audio(path, sample_rate=16000):
    """Decode MP3/FLAC audio as a mono float tensor."""
    samples, source_sample_rate = sf.read(
        str(path),
        dtype="float32",
        always_2d=True,
    )
    waveform = torch.from_numpy(samples).transpose(0, 1).mean(dim=0)

    if source_sample_rate != sample_rate:
        waveform = audio_functional.resample(
            waveform,
            source_sample_rate,
            sample_rate,
        )

    return waveform


def _hz_to_mel(frequency):
    return 2595.0 * math.log10(1.0 + frequency / 700.0)


def _mel_to_hz(mel):
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


@lru_cache(maxsize=16)
def mel_filterbank(sample_rate, n_fft, n_mels, f_min, f_max):
    max_frequency = f_max or sample_rate / 2
    min_mel = _hz_to_mel(f_min)
    max_mel = _hz_to_mel(max_frequency)
    mel_points = torch.linspace(min_mel, max_mel, n_mels + 2)
    hz_points = torch.tensor([_mel_to_hz(value.item()) for value in mel_points])
    bins = torch.floor((n_fft + 1) * hz_points / sample_rate).long()

    filterbank = torch.zeros(n_mels, n_fft // 2 + 1)
    max_bin = n_fft // 2

    for mel_index in range(n_mels):
        left = int(bins[mel_index].clamp(0, max_bin))
        center = int(bins[mel_index + 1].clamp(0, max_bin))
        right = int(bins[mel_index + 2].clamp(0, max_bin))

        if center <= left:
            center = min(left + 1, max_bin)
        if right <= center:
            right = min(center + 1, max_bin)

        if center > left:
            filterbank[mel_index, left:center] = (
                torch.arange(left, center) - left
            ) / (center - left)
        if right > center:
            filterbank[mel_index, center:right] = (
                right - torch.arange(center, right)
            ) / (right - center)

    return filterbank


def log_mel_spectrogram(
    waveform,
    sample_rate=16000,
    n_fft=400,
    hop_length=160,
    win_length=400,
    n_mels=80,
    f_min=20.0,
    f_max=None,
):
    """Create a normalized log-Mel spectrogram shaped [mel, time]."""
    if waveform.numel() < win_length:
        waveform = F.pad(waveform, (0, win_length - waveform.numel()))

    window = torch.hann_window(win_length)
    spectrum = torch.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        return_complex=True,
    )
    power = spectrum.abs().pow(2)
    filters = mel_filterbank(
        sample_rate,
        n_fft,
        n_mels,
        float(f_min),
        float(f_max) if f_max else None,
    )
    mel = filters @ power
    log_mel = torch.log(mel.clamp_min(1e-10))
    mean = log_mel.mean(dim=1, keepdim=True)
    std = log_mel.std(dim=1, keepdim=True).clamp_min(1e-5)
    return (log_mel - mean) / std
