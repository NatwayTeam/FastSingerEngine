import argparse
from collections import OrderedDict, defaultdict
import json
import math
import os
import random
import re

import librosa
from librosa.filters import mel as librosa_mel_fn
from librosa.util import pad_center
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy.signal import get_window
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils import weight_norm, remove_weight_norm
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import yaml

matplotlib.use("Agg")

phones = [
    "a", "ai", "an", "ang", "ao", "b", "c", "ch", "d", "e", "ei", "en",
    "eng", "er", "f", "g", "h", "i", "ia", "ian", "iang", "iao", "ie", "in",
    "ing", "iong", "iu", "j", "k", "l", "m", "n", "o", "ong", "ou", "p", "q",
    "r", "s", "sh", "t", "u", "ua", "uai", "uan", "uang", "uei", "uen", "uo",
    "v", "ve", "w", "x", "y", "z", "zh",
]
PAD = 0

symbols = ["_pad"] + phones
symbol_to_id = {s: i for i, s in enumerate(symbols)}

VOICELESS_INITIALS = {
    "b", "c", "ch", "d", "f", "g", "h", "j", "k", "p", "q", "s", "sh",
    "t", "x", "z", "zh",
}

INITIALS_2 = ("zh", "ch", "sh")
INITIALS_1 = tuple("bpmfdtnlgkhjqxrzcs")
Y_JQX = ("y", "j", "q", "x")
FINAL_FIX = {"ui": "uei", "un": "uen", "ue": "ve"}


def pinyin_to_phones(pinyin):
    """Derive the FastSinger phone sequence from a pinyin spelling.

    Purely spelling based, so preprocessing never depends on a hand written
    transcript file.
    """
    if pinyin.startswith(INITIALS_2):
        initial, rest = pinyin[:2], pinyin[2:]
    elif pinyin.startswith(INITIALS_1):
        initial, rest = pinyin[:1], pinyin[1:]
    elif pinyin.startswith(("y", "w")):
        initial, rest = pinyin[:1], pinyin[1:]
    else:
        return [pinyin]

    if rest in FINAL_FIX:
        final = FINAL_FIX[rest]
    elif initial in Y_JQX and rest == "u":
        final = "v"
    else:
        final = rest
    return [initial, final]


def phone_to_sequence(text):
    tokens = text.replace("{", " ").replace("}", " ").split()
    return [symbol_to_id[t] for t in tokens if t in symbol_to_id]


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


class STFT(torch.nn.Module):
    def __init__(self, filter_length, hop_length, win_length, window="hann"):
        super(STFT, self).__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        scale = self.filter_length / self.hop_length
        fourier_basis = np.fft.fft(np.eye(self.filter_length))

        cutoff = int((self.filter_length / 2 + 1))
        fourier_basis = np.vstack(
            [np.real(fourier_basis[:cutoff, :]), np.imag(fourier_basis[:cutoff, :])]
        )

        forward_basis = torch.FloatTensor(fourier_basis[:, None, :])
        inverse_basis = torch.FloatTensor(
            np.linalg.pinv(scale * fourier_basis).T[:, None, :]
        )

        if window is not None:
            assert filter_length >= win_length
            fft_window = get_window(window, win_length, fftbins=True)
            fft_window = pad_center(fft_window, size=filter_length)
            fft_window = torch.from_numpy(fft_window).float()

            forward_basis *= fft_window
            inverse_basis *= fft_window

        self.register_buffer("forward_basis", forward_basis.float())
        self.register_buffer("inverse_basis", inverse_basis.float())

    def transform(self, input_data):
        num_batches = input_data.size(0)
        num_samples = input_data.size(1)

        input_data = input_data.view(num_batches, 1, num_samples)
        input_data = F.pad(
            input_data.unsqueeze(1),
            (int(self.filter_length / 2), int(self.filter_length / 2), 0, 0),
            mode="reflect",
        )
        input_data = input_data.squeeze(1)

        forward_transform = F.conv1d(
            input_data.to(self.forward_basis.device),
            torch.autograd.Variable(self.forward_basis, requires_grad=False),
            stride=self.hop_length,
            padding=0,
        ).cpu()

        cutoff = int((self.filter_length / 2) + 1)
        real_part = forward_transform[:, :cutoff, :]
        imag_part = forward_transform[:, cutoff:, :]

        magnitude = torch.sqrt(real_part ** 2 + imag_part ** 2)
        phase = torch.autograd.Variable(torch.atan2(imag_part.data, real_part.data))

        return magnitude, phase


class LogMelSTFT(torch.nn.Module):
    def __init__(
        self,
        filter_length,
        hop_length,
        win_length,
        n_mel_channels,
        sampling_rate,
        mel_fmin,
        mel_fmax,
    ):
        super(LogMelSTFT, self).__init__()
        self.n_mel_channels = n_mel_channels
        self.sampling_rate = sampling_rate
        self.stft_fn = STFT(filter_length, hop_length, win_length)
        mel_basis = librosa_mel_fn(
            sr=sampling_rate,
            n_fft=filter_length,
            n_mels=n_mel_channels,
            fmin=mel_fmin,
            fmax=mel_fmax,
        )
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer("mel_basis", mel_basis)

    def spectral_normalize(self, magnitudes):
        output = dynamic_range_compression(magnitudes)
        return output

    def mel_spectrogram(self, y):
        assert torch.min(y.data) >= -1
        assert torch.max(y.data) <= 1

        magnitudes, phases = self.stft_fn.transform(y)
        magnitudes = magnitudes.data
        mel_output = torch.matmul(self.mel_basis, magnitudes)
        mel_output = self.spectral_normalize(mel_output)
        energy = torch.norm(magnitudes, dim=1)

        return mel_output, energy


def get_mel_from_wav(audio, _stft):
    audio = torch.clip(torch.FloatTensor(audio).unsqueeze(0), -1, 1)
    audio = torch.autograd.Variable(audio, requires_grad=False)
    melspec, energy = _stft.mel_spectrogram(audio)
    melspec = torch.squeeze(melspec, 0).numpy().astype(np.float32)
    energy = torch.squeeze(energy, 0).numpy().astype(np.float32)

    return melspec, energy


def find_initial_boundary(mel, frac=0.8, sustain=2, search=15):
    """Locate the initial -> final frame boundary from a mel spectrogram.

    The final of an isolated syllable is the vowel, so the boundary is the
    first frame whose spectrum has settled onto the vowel: its distance to
    the second-half reference drops to `frac` of the peak distance and stays
    there for `sustain` frames. If it never settles inside `search` frames,
    fall back to the steepest spectral descent within twice that window.
    """
    z = (mel - mel.mean(axis=1, keepdims=True)) / (
        mel.std(axis=1, keepdims=True) + 1e-6
    )
    total = z.shape[1]
    if total < 4:
        return 1

    ref = z[:, total // 2 :].mean(axis=1)
    dist = np.linalg.norm(z - ref[:, None], axis=0)
    if total >= 3:
        dist = np.convolve(dist, np.ones(3) / 3.0, mode="same")

    hi = min(search, total)
    below = dist[:hi] <= dist[:hi].max() * frac
    for i in range(0, hi - sustain + 1):
        if below[i : i + sustain].all():
            return int(np.clip(i, 1, total - 1))

    hi = min(2 * search, total - 1)
    if hi > 1:
        drop = dist[1:hi] - dist[2 : hi + 1]
        return int(np.clip(int(np.argmax(drop)) + 1, 1, total - 1))
    return 1


class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)
        output = torch.bmm(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)

        mask = mask.repeat(n_head, 1, 1)
        output, attn = self.attention(q, k, v, mask=mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output, attn


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_in, d_hid, kernel_size, dropout=0.1):
        super().__init__()

        self.w_1 = nn.Conv1d(
            d_in,
            d_hid,
            kernel_size=kernel_size[0],
            padding=(kernel_size[0] - 1) // 2,
        )
        self.w_2 = nn.Conv1d(
            d_hid,
            d_in,
            kernel_size=kernel_size[1],
            padding=(kernel_size[1] - 1) // 2,
        )

        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)

        return output


class FFTBlock(torch.nn.Module):
    def __init__(self, d_model, n_head, d_k, d_v, d_inner, kernel_size, dropout=0.1):
        super(FFTBlock, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, kernel_size, dropout=dropout
        )

    def forward(self, enc_input, mask=None, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask
        )
        enc_output = enc_output.masked_fill(mask.unsqueeze(-1), 0)

        enc_output = self.pos_ffn(enc_output)
        enc_output = enc_output.masked_fill(mask.unsqueeze(-1), 0)

        return enc_output, enc_slf_attn


class ConvNorm(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=None,
        dilation=1,
        bias=True,
    ):
        super(ConvNorm, self).__init__()

        if padding is None:
            assert kernel_size % 2 == 1
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, signal):
        conv_signal = self.conv(signal)

        return conv_signal


class PostNet(nn.Module):
    def __init__(
        self,
        n_mel_channels=80,
        postnet_embedding_dim=512,
        postnet_kernel_size=5,
        postnet_n_convolutions=5,
    ):
        super(PostNet, self).__init__()
        self.convolutions = nn.ModuleList()

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(
                    n_mel_channels,
                    postnet_embedding_dim,
                    kernel_size=postnet_kernel_size,
                    stride=1,
                    padding=int((postnet_kernel_size - 1) / 2),
                    dilation=1,
                ),
                nn.BatchNorm1d(postnet_embedding_dim),
            )
        )

        for i in range(1, postnet_n_convolutions - 1):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(
                        postnet_embedding_dim,
                        postnet_embedding_dim,
                        kernel_size=postnet_kernel_size,
                        stride=1,
                        padding=int((postnet_kernel_size - 1) / 2),
                        dilation=1,
                    ),
                    nn.BatchNorm1d(postnet_embedding_dim),
                )
            )

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(
                    postnet_embedding_dim,
                    n_mel_channels,
                    kernel_size=postnet_kernel_size,
                    stride=1,
                    padding=int((postnet_kernel_size - 1) / 2),
                    dilation=1,
                ),
                nn.BatchNorm1d(n_mel_channels),
            )
        )

    def forward(self, x):
        x = x.contiguous().transpose(1, 2)

        for i in range(len(self.convolutions) - 1):
            x = F.dropout(torch.tanh(self.convolutions[i](x)), 0.5, self.training)
        x = F.dropout(self.convolutions[-1](x), 0.5, self.training)

        x = x.contiguous().transpose(1, 2)
        return x


def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array(
        [get_posi_angle_vec(pos_i) for pos_i in range(n_position)]
    )

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])

    if padding_idx is not None:
        sinusoid_table[padding_idx] = 0.0

    return torch.FloatTensor(sinusoid_table)


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()

        n_position = config["max_seq_len"] + 1
        n_src_vocab = len(symbols)
        d_word_vec = config["transformer"]["encoder_hidden"]
        n_layers = config["transformer"]["encoder_layer"]
        n_head = config["transformer"]["encoder_head"]
        d_k = d_v = (
            config["transformer"]["encoder_hidden"]
            // config["transformer"]["encoder_head"]
        )
        d_model = config["transformer"]["encoder_hidden"]
        d_inner = config["transformer"]["conv_filter_size"]
        kernel_size = config["transformer"]["conv_kernel_size"]
        dropout = config["transformer"]["encoder_dropout"]

        self.max_seq_len = config["max_seq_len"]
        self.d_model = d_model

        self.src_word_emb = nn.Embedding(
            n_src_vocab, d_word_vec, padding_idx=PAD
        )
        self.position_enc = nn.Parameter(
            get_sinusoid_encoding_table(n_position, d_word_vec).unsqueeze(0),
            requires_grad=False,
        )

        self.layer_stack = nn.ModuleList(
            [
                FFTBlock(
                    d_model, n_head, d_k, d_v, d_inner, kernel_size, dropout=dropout
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, src_seq, mask, return_attns=False):
        enc_slf_attn_list = []
        batch_size, max_len = src_seq.shape[0], src_seq.shape[1]

        slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)

        if not self.training and src_seq.shape[1] > self.max_seq_len:
            enc_output = self.src_word_emb(src_seq) + get_sinusoid_encoding_table(
                src_seq.shape[1], self.d_model
            )[: src_seq.shape[1], :].unsqueeze(0).expand(batch_size, -1, -1).to(
                src_seq.device
            )
        else:
            enc_output = self.src_word_emb(src_seq) + self.position_enc[
                :, :max_len, :
            ].expand(batch_size, -1, -1)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output, mask=mask, slf_attn_mask=slf_attn_mask
            )
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        return enc_output


LRELU_SLOPE = 0.1


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


class ResBlock(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(ResBlock, self).__init__()
        self.convs1 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[0],
                        padding=get_padding(kernel_size, dilation[0]),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[1],
                        padding=get_padding(kernel_size, dilation[1]),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[2],
                        padding=get_padding(kernel_size, dilation[2]),
                    )
                ),
            ]
        )
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
            ]
        )
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class Vocoder(torch.nn.Module):
    def __init__(self, h):
        super(Vocoder, self).__init__()
        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)
        self.conv_pre = weight_norm(
            Conv1d(80, h.upsample_initial_channel, 7, 1, padding=3)
        )
        resblock = ResBlock

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            self.ups.append(
                weight_norm(
                    ConvTranspose1d(
                        h.upsample_initial_channel // (2 ** i),
                        h.upsample_initial_channel // (2 ** (i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2,
                    )
                )
            )

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = h.upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(
                zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes)
            ):
                self.resblocks.append(resblock(ch, k, d))

        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, x):
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VarianceAdaptor(nn.Module):
    def __init__(self, preprocess_config, model_config):
        super(VarianceAdaptor, self).__init__()
        self.length_regulator = LengthRegulator()
        self.energy_predictor = VariancePredictor(model_config)

        pitch_quantization = model_config["variance_embedding"]["pitch_quantization"]
        energy_quantization = model_config["variance_embedding"]["energy_quantization"]
        n_bins = model_config["variance_embedding"]["n_bins"]
        assert pitch_quantization in ["linear", "log"]
        assert energy_quantization in ["linear", "log"]
        with open(
            os.path.join(preprocess_config["path"]["preprocessed_path"], "stats.json")
        ) as f:
            stats = json.load(f)
            pitch_min, pitch_max = stats["pitch"][:2]
            energy_min, energy_max = stats["energy"][:2]

        if pitch_quantization == "log":
            self.pitch_bins = nn.Parameter(
                torch.exp(
                    torch.linspace(np.log(pitch_min), np.log(pitch_max), n_bins - 1)
                ),
                requires_grad=False,
            )
        else:
            self.pitch_bins = nn.Parameter(
                torch.linspace(pitch_min, pitch_max, n_bins - 1),
                requires_grad=False,
            )
        if energy_quantization == "log":
            self.energy_bins = nn.Parameter(
                torch.exp(
                    torch.linspace(np.log(energy_min), np.log(energy_max), n_bins - 1)
                ),
                requires_grad=False,
            )
        else:
            self.energy_bins = nn.Parameter(
                torch.linspace(energy_min, energy_max, n_bins - 1),
                requires_grad=False,
            )

        self.pitch_embedding = nn.Embedding(
            n_bins, model_config["transformer"]["encoder_hidden"]
        )
        self.energy_embedding = nn.Embedding(
            n_bins, model_config["transformer"]["encoder_hidden"]
        )

    def get_energy_embedding(self, x, target, mask, control):
        prediction = self.energy_predictor(x, mask)
        if target is not None:
            embedding = self.energy_embedding(torch.bucketize(target, self.energy_bins))
        else:
            prediction = prediction * control
            embedding = self.energy_embedding(
                torch.bucketize(prediction, self.energy_bins)
            )
        return prediction, embedding

    def forward(
        self,
        x,
        src_mask,
        mel_mask=None,
        max_len=None,
        f0=None,
        energy_target=None,
        duration_target=None,
        e_control=1.0,
    ):
        assert f0 is not None, "f0 must be provided"
        assert duration_target is not None, "duration must be provided"

        energy_prediction, energy_embedding = self.get_energy_embedding(
            x, energy_target, src_mask, e_control
        )
        x = x + energy_embedding

        x, mel_len = self.length_regulator(x, duration_target, max_len)
        duration_rounded = duration_target
        if mel_mask is None:
            mel_mask = get_mask_from_lengths(mel_len)

        pitch_embedding = self.pitch_embedding(torch.bucketize(f0, self.pitch_bins))
        x = x + pitch_embedding

        return (
            x,
            energy_prediction,
            duration_rounded,
            mel_len,
            mel_mask,
        )


class LengthRegulator(nn.Module):
    def __init__(self):
        super(LengthRegulator, self).__init__()

    def LR(self, x, duration, max_len):
        output = list()
        mel_len = list()
        for batch, expand_target in zip(x, duration):
            expanded = self.expand(batch, expand_target)
            output.append(expanded)
            mel_len.append(expanded.shape[0])

        if max_len is not None:
            output = pad(output, max_len)
        else:
            output = pad(output)

        return output, torch.LongTensor(mel_len, device=x.device)

    def expand(self, batch, predicted):
        out = list()

        for i, vec in enumerate(batch):
            expand_size = predicted[i].item()
            out.append(vec.expand(max(int(expand_size), 0), -1))
        out = torch.cat(out, 0)

        return out

    def forward(self, x, duration, max_len):
        output, mel_len = self.LR(x, duration, max_len)
        return output, mel_len


class VariancePredictor(nn.Module):
    def __init__(self, model_config):
        super(VariancePredictor, self).__init__()

        self.input_size = model_config["transformer"]["encoder_hidden"]
        self.filter_size = model_config["variance_predictor"]["filter_size"]
        self.kernel = model_config["variance_predictor"]["kernel_size"]
        self.conv_output_size = model_config["variance_predictor"]["filter_size"]
        self.dropout = model_config["variance_predictor"]["dropout"]

        self.conv_layer = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv1d_1",
                        Conv(
                            self.input_size,
                            self.filter_size,
                            kernel_size=self.kernel,
                            padding=(self.kernel - 1) // 2,
                        ),
                    ),
                    ("relu_1", nn.ReLU()),
                    ("layer_norm_1", nn.LayerNorm(self.filter_size)),
                    ("dropout_1", nn.Dropout(self.dropout)),
                    (
                        "conv1d_2",
                        Conv(
                            self.filter_size,
                            self.filter_size,
                            kernel_size=self.kernel,
                            padding=1,
                        ),
                    ),
                    ("relu_2", nn.ReLU()),
                    ("layer_norm_2", nn.LayerNorm(self.filter_size)),
                    ("dropout_2", nn.Dropout(self.dropout)),
                ]
            )
        )

        self.linear_layer = nn.Linear(self.conv_output_size, 1)

    def forward(self, encoder_output, mask):
        out = self.conv_layer(encoder_output)
        out = self.linear_layer(out)
        out = out.squeeze(-1)

        if mask is not None:
            out = out.masked_fill(mask, 0.0)

        return out


class Conv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
    ):
        super(Conv, self).__init__()

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, x):
        x = x.contiguous().transpose(1, 2)
        x = self.conv(x)
        x = x.contiguous().transpose(1, 2)

        return x


class FastSinger(nn.Module):
    def __init__(self, preprocess_config, model_config):
        super(FastSinger, self).__init__()
        self.model_config = model_config

        self.encoder = Encoder(model_config)
        self.variance_adaptor = VarianceAdaptor(preprocess_config, model_config)
        self.mel_linear = nn.Linear(
            model_config["transformer"]["encoder_hidden"],
            preprocess_config["audio"]["n_mel_channels"],
        )
        self.postnet = PostNet()

    def forward(
        self,
        texts,
        src_lens,
        max_src_len,
        mels=None,
        mel_lens=None,
        max_mel_len=None,
        f0s=None,
        e_targets=None,
        d_targets=None,
        e_control=1.0,
    ):
        src_masks = get_mask_from_lengths(src_lens, max_src_len)
        mel_masks = (
            get_mask_from_lengths(mel_lens, max_mel_len)
            if mel_lens is not None
            else None
        )

        output = self.encoder(texts, src_masks)

        (
            output,
            e_predictions,
            d_rounded,
            mel_lens,
            mel_masks,
        ) = self.variance_adaptor(
            output,
            src_masks,
            mel_masks,
            max_mel_len,
            f0s,
            e_targets,
            d_targets,
            e_control,
        )

        output = self.mel_linear(output)

        postnet_output = self.postnet(output) + output

        return (
            output,
            postnet_output,
            e_predictions,
            d_rounded,
            src_masks,
            mel_masks,
            src_lens,
            mel_lens,
        )


class FastSingerLoss(nn.Module):
    def __init__(self, preprocess_config, model_config):
        super(FastSingerLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()

    def forward(self, inputs, predictions):
        (
            mel_targets,
            _,
            _,
            _,
            energy_targets,
            _,
        ) = inputs[5:]
        (
            mel_predictions,
            postnet_mel_predictions,
            energy_predictions,
            _,
            src_masks,
            mel_masks,
            _,
            _,
        ) = predictions
        src_masks = ~src_masks
        mel_masks = ~mel_masks
        mel_targets = mel_targets[:, : mel_masks.shape[1], :]
        mel_masks = mel_masks[:, :mel_masks.shape[1]]

        energy_predictions = energy_predictions.masked_select(src_masks)
        energy_targets = energy_targets.masked_select(src_masks)

        mel_predictions = mel_predictions.masked_select(mel_masks.unsqueeze(-1))
        postnet_mel_predictions = postnet_mel_predictions.masked_select(
            mel_masks.unsqueeze(-1)
        )
        mel_targets = mel_targets.masked_select(mel_masks.unsqueeze(-1))

        mel_loss = self.mae_loss(mel_predictions, mel_targets)
        postnet_mel_loss = self.mae_loss(postnet_mel_predictions, mel_targets)
        energy_loss = self.mse_loss(energy_predictions, energy_targets)

        total_loss = mel_loss + postnet_mel_loss + energy_loss

        return (
            total_loss,
            mel_loss,
            postnet_mel_loss,
            energy_loss,
        )


def build_phone_frames(texts, text_lens, durations, frames):
    """Expand per-phone durations into a per-frame phone label map.

    Lets us ask "does this mel actually contain the phone we asked for, at the
    right moment" with a plain cross-entropy instead of a content
    discriminator, which this architecture does not otherwise need (text reaches
    the decoder through an explicit path, not a bottleneck).
    """
    texts = torch.as_tensor(texts)
    durations = torch.as_tensor(durations)
    B, L = texts.shape
    out = torch.zeros(B, frames, dtype=torch.long, device=texts.device)
    for b in range(B):
        pos = 0
        for j in range(int(text_lens[b])):
            n = int(durations[b, j])
            if n <= 0:
                continue
            out[b, pos:pos + n] = texts[b, j]
            pos += n
    return out


def pool_phone_targets(phone_labels, valid, n_classes, t_out):
    """Pool a per-frame phone map down to the classifier's time resolution.

    The classifier downsamples time, so the label map has to be pooled as well.
    Truncating it instead would compare a short label map against a long logit
    map, which either crashes or silently disables the term entirely.
    """
    lab = F.adaptive_avg_pool1d(
        F.one_hot(phone_labels, num_classes=n_classes).permute(0, 2, 1).float(),
        t_out,
    ).argmax(1)
    mask = F.adaptive_max_pool1d(valid.float().unsqueeze(1), t_out).squeeze(1)
    return lab, mask


def pitch_to_channel(pitch):
    """Map frame-level f0 (Hz) to a roughly [-1, 1] channel for the discriminator.

    440 Hz -> 0, one octave -> +-0.625. Unvoiced frames (f0 == 0) go to -2 so the
    discriminator can tell "no pitch" from "very low pitch".
    """
    semitone = 12.0 * torch.log2(pitch.clamp(min=1.0) / 440.0)
    ch = semitone / 19.2
    return torch.where(pitch > 0, ch, torch.full_like(ch, -2.0)).unsqueeze(1)


class MelDiscriminator(nn.Module):
    """Multi-scale 2D conv discriminator over a mel spectrogram.

    Operates on mel rather than waveform on purpose: FastSinger's output is a
    mel and the vocoder is frozen, so a waveform discriminator would train
    against a module that is not being optimised. Scales are cheap average
    pools along the time axis.
    """

    def __init__(self, in_channels, base=32, n_scales=3):
        super(MelDiscriminator, self).__init__()
        self.n_scales = n_scales
        self.scales = nn.ModuleList()
        for _ in range(n_scales):
            layers = []
            c_in = in_channels
            spec = [(base, 2), (base * 2, 2), (base * 2, 2), (base * 4, 1)]
            for i, (c_out, stride) in enumerate(spec):
                layers.append(
                    nn.Conv2d(c_in, c_out, (3, 9), (1, stride), padding=(1, 4),
                              bias=False)
                )
                if i > 0:
                    layers.append(nn.BatchNorm2d(c_out))
                layers.append(nn.LeakyReLU(0.1, inplace=True))
                c_in = c_out
            layers.append(nn.Conv2d(c_in, 1, (3, 3), (1, 1), padding=(1, 1)))
            self.scales.append(nn.Sequential(*layers))

    def forward(self, x):
        """x: (B, C, T). Returns per-scale logits and feature maps."""
        logits, feats = [], []
        for i, net in enumerate(self.scales):
            xi = x.unsqueeze(2)
            xi = F.avg_pool2d(xi, (1, 2 ** i)) if i else xi
            h = xi
            for layer in net:
                h = layer(h)
                if isinstance(layer, nn.LeakyReLU) and h.shape[1] > 1:
                    feats.append(h)
            logits.append(h)
        return logits, feats


class PhoneClassifier(nn.Module):
    """Per-frame phone posterior over the generated mel (the "semantic" term)."""

    def __init__(self, n_phones, n_mel_channels=80, base=32):
        super(PhoneClassifier, self).__init__()
        spec = [(base, 2), (base * 2, 2), (base * 2, 2), (base * 2, 1)]
        layers, c_in = [], n_mel_channels
        for c_out, stride in spec:
            layers.append(
                nn.Conv2d(c_in, c_out, (3, 9), (1, stride), padding=(1, 4), bias=False)
            )
            layers.append(nn.BatchNorm2d(c_out))
            layers.append(nn.LeakyReLU(0.1, inplace=True))
            c_in = c_out
        layers.append(nn.Conv2d(c_in, n_phones, (3, 3), (1, 1), padding=(1, 1)))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """x: (B, C, T) -> per-frame logits (B, n_phones, T')."""
        return self.net(x.unsqueeze(2)).squeeze(2)


def hinge_d_loss(logits_real, logits_fake):
    loss = 0.0
    for lr_, lf_ in zip(logits_real, logits_fake):
        loss = loss + F.relu(1.0 - lr_).mean() + F.relu(1.0 + lf_).mean()
    return loss / max(len(logits_real), 1)


def hinge_g_loss(logits_fake):
    loss = 0.0
    for lf_ in logits_fake:
        loss = loss + F.relu(1.0 - lf_).mean()
    return loss / max(len(logits_fake), 1)


def feature_matching_loss(feats_real, feats_fake):
    if not feats_real:
        return torch.zeros((), device=feats_fake[0].device)
    loss = 0.0
    for fr, ff in zip(feats_real, feats_fake):
        if fr.shape != ff.shape:
            continue
        loss = loss + F.l1_loss(ff, fr.detach())
    return loss / max(len(feats_fake), 1)


class Adversarial(nn.Module):
    """Three training-time regularisers on the generated mel.

    realism  : multi-scale mel discriminator (hinge) + feature matching
    semantics: per-frame phone cross-entropy against the known duration map
    pitch    : discriminator on (mel, input f0), so harmonics have to line up
               with the pitch the caller asked for
    """

    def __init__(self, preprocess_config, model_config, train_config):
        super(Adversarial, self).__init__()
        cfg = train_config.get("adversarial") or {}
        self.enabled = bool(cfg.get("enabled", False))
        self.start_step = int(cfg.get("start_step", 3000))
        self.ramp_steps = max(1, int(cfg.get("ramp_steps", 1000)))
        self.w_adv = float(cfg.get("adv_weight", 0.05))
        self.w_fm = float(cfg.get("fm_weight", 0.2))
        self.w_phone = float(cfg.get("phone_weight", 0.1))
        self.w_pitch = float(cfg.get("pitch_weight", 0.05))
        self.d_iters = int(cfg.get("d_iters", 1))
        self.d_lr = float(cfg.get("d_lr", 1e-4))
        self.d_min_lr = float(cfg.get("d_min_lr", self.d_lr * 0.01))
        # 判别器跟生成器用同一条 warmup + 余弦曲线, 只是峰值不同。让 G 衰减到
        # 接近 0 而 D 保持恒定学习率, 会让训练后期判别器稳赢, 对抗项反而
        # 变成主导项, 这正是 hi_ceiling 想避免的情形。
        self.d_warmup_step = int((train_config.get("optimizer") or {}).get("warmup_step", 0))
        self.d_total_step = int((train_config.get("step") or {}).get("total_step", 0))
        self.d_current_lr = self.d_lr
        base = int(cfg.get("base_channels", 32))
        n_mel = int(preprocess_config["audio"]["n_mel_channels"])

        # Dynamic gain on the adversarial terms. Two failure modes have to be
        # traded off and they pull in opposite directions, so a fixed weight
        # cannot serve both:
        #   * discriminator too strong -> the generator chases artifacts instead
        #     of learning the data, which sounds like injected hiss
        #   * discriminator saturated  -> its gradient carries no information and
        #     it just perturbs a generator that was fine on plain L1
        # `gain` multiplies the adversarial terms only; the phone term is a
        # supervised anchor and deliberately stays fixed.
        self.auto_gain = bool(cfg.get("auto_gain", True))
        self.gain = 1.0
        self.gain_min = float(cfg.get("gain_min", 0.2)) / max(self.w_adv, 1e-8)
        self.gain_max = float(cfg.get("gain_max", 3.0)) / max(self.w_adv, 1e-8)
        self.eq_target = float(cfg.get("eq_target", 0.6))
        self.ctrl_k = float(cfg.get("ctrl_k", 0.05))
        self.ctrl_every = max(1, int(cfg.get("ctrl_every", 25)))
        self.hi_bands = max(1, int(cfg.get("hi_bands", 12)))
        self.hi_ceiling = float(cfg.get("hi_ceiling", 0.2))
        self._gap_ema = None
        self._hi_ema = None

        self.d_adv = MelDiscriminator(n_mel, base=base)
        self.d_pitch = MelDiscriminator(n_mel + 1, base=base)
        self.classifier = PhoneClassifier(
            len(symbols), n_mel_channels=n_mel, base=base
        )
        params = list(self.d_adv.parameters()) + list(self.d_pitch.parameters())
        params += list(self.classifier.parameters())
        self.d_optimizer = torch.optim.Adam(params, lr=self.d_lr, betas=(0.5, 0.9))

        # The classifier is trained on *real* mels only, so it never becomes an
        # adversary -- but if it keeps getting better while the generator chases
        # it, the phone term is a moving target with no fixed optimum and the
        # generator ends up chasing a label boundary instead of the data. So it
        # gets a warmup window and is then frozen, which is what "supervised
        # anchor" is supposed to mean. The freeze point is derived from the
        # absolute step, so a resumed run freezes at the same place.
        cls_until = cfg.get("cls_train_until")
        self.cls_train_until = (
            int(cls_until) if cls_until is not None else self.start_step + self.ramp_steps
        )
        self.cls_ce_weight = float(cfg.get("cls_ce_weight", 0.1))
        self.cls_frozen = False

    def _set_d_grad(self, flag):
        for p in self.d_adv.parameters():
            p.requires_grad_(flag)
        for p in self.d_pitch.parameters():
            p.requires_grad_(flag)
        for p in self.classifier.parameters():
            p.requires_grad_(flag and not self.cls_frozen)

    def freeze_classifier(self):
        """Stop updating the phone classifier so the semantic term is stationary.

        Idempotent, and safe to call every step: the condition is a pure
        function of the global step, so it holds across resumes.
        """
        if self.cls_frozen:
            return
        self.cls_frozen = True
        for p in self.classifier.parameters():
            p.requires_grad_(False)

    def set_step(self, step):
        """把判别器的 lr 推到与生成器同步的调度点上。"""
        self.d_current_lr = cosine_lr(
            step, self.d_lr, self.d_min_lr, self.d_warmup_step, self.d_total_step
        )
        for param_group in self.d_optimizer.param_groups:
            param_group["lr"] = self.d_current_lr
        return self.d_current_lr

    def scale(self, step):
        """Ramp the adversarial terms in instead of switching them on at full
        weight.

        A generator that has only ever seen L1 suddenly receives a large
        adversarial gradient and the usual outcome is not extra detail but
        high-frequency junk that the (strong) discriminator happens to accept.
        """
        return min(1.0, max(0.0, (step - self.start_step + 1) / float(self.ramp_steps)))

    def update_gain(self, d_gap, hi_gap):
        """Closed-loop control of the adversarial weight.

        d_gap = E[D(fake)] - E[D(real)] under hinge. 0 means neither side is
        winning; about -2 means the discriminator separates real from fake
        trivially. hi_gap is the fake-minus-real energy in the top mel bands,
        which is what injected hiss looks like numerically.
        """
        self._gap_ema = d_gap if self._gap_ema is None else 0.98 * self._gap_ema + 0.02 * d_gap
        self._hi_ema = hi_gap if self._hi_ema is None else 0.98 * self._hi_ema + 0.02 * hi_gap
        strength = max(0.0, -self._gap_ema)

        if self._hi_ema > self.hi_ceiling:
            # brake hard: the generator is buying acceptance with top-band energy
            self.gain *= 0.97
        else:
            # Proportional, so the gain settles instead of oscillating between
            # the rails: at strength == eq_target the factor is exactly 1.
            self.gain *= math.exp(-self.ctrl_k * (strength - self.eq_target))
        self.gain = min(self.gain_max, max(self.gain_min, self.gain))
        return self._gap_ema, self._hi_ema

    def d_step(self, mel_real, mel_fake, pitch, phone_labels, step):
        """One or more discriminator updates. Inputs are (B, T, C), detached.

        `step` is required, not optional: the classifier freeze point is derived
        from the global step, so omitting it would silently disable freezing and
        put the phone term back on a moving target.
        """
        if step >= self.cls_train_until:
            self.freeze_classifier()

        x_real = mel_real.transpose(1, 2)
        x_fake = mel_fake.transpose(1, 2)
        xp_real = torch.cat([x_real, pitch_to_channel(pitch)], dim=1)
        xp_fake = torch.cat([x_fake, pitch_to_channel(pitch)], dim=1)

        self._set_d_grad(True)
        d_loss = torch.zeros((), device=mel_real.device)
        gap = torch.zeros((), device=mel_real.device)
        for _ in range(max(self.d_iters, 0)):
            self.d_optimizer.zero_grad()
            lr_adv, _ = self.d_adv(x_real)
            lf_adv, _ = self.d_adv(x_fake.detach())
            lr_pit, _ = self.d_pitch(xp_real)
            lf_pit, _ = self.d_pitch(xp_fake.detach())
            loss = hinge_d_loss(lr_adv, lf_adv) + hinge_d_loss(lr_pit, lf_pit)

            if not self.cls_frozen:
                cls_real = self.classifier(x_real.detach()).float()
                if cls_real.shape[-1] >= 4:
                    lab, cmask = pool_phone_targets(
                        phone_labels, torch.ones_like(pitch, dtype=torch.bool),
                        cls_real.shape[1], cls_real.shape[-1]
                    )
                    ce = F.cross_entropy(cls_real, lab, reduction="none")
                    loss = loss + self.cls_ce_weight * (ce * cmask).sum() \
                        / cmask.sum().clamp(min=1.0)

            d_loss = loss
            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 5.0)
            self.d_optimizer.step()

            r = torch.cat([t.flatten() for t in lr_adv + lr_pit]).mean()
            f = torch.cat([t.flatten() for t in lf_adv + lf_pit]).mean()
            gap = (f - r).detach()
        self._set_d_grad(False)
        return d_loss.detach(), gap

    def g_terms(self, mel_real, mel_fake, pitch, phone_labels, valid):
        """Adversarial / semantic / pitch terms added to the generator loss."""
        self._set_d_grad(False)
        x_real = mel_real.transpose(1, 2)
        x_fake = mel_fake.transpose(1, 2)
        xp_fake = torch.cat([x_fake, pitch_to_channel(pitch)], dim=1)
        xp_real = torch.cat([x_real, pitch_to_channel(pitch)], dim=1)

        lf_adv, ff_adv = self.d_adv(x_fake)
        _, fr_adv = self.d_adv(x_real)
        adv = hinge_g_loss(lf_adv)
        fm = feature_matching_loss(fr_adv, ff_adv)

        lf_pit, ff_pit = self.d_pitch(xp_fake)
        _, fr_pit = self.d_pitch(xp_real)
        pitch_adv = hinge_g_loss(lf_pit)
        pitch_fm = feature_matching_loss(fr_pit, ff_pit)

        logits = self.classifier(x_fake).float()
        # Pool (not truncate) the per-frame phone map to the classifier's rate.
        if logits.shape[-1] >= 4:
            lab, mask = pool_phone_targets(
                phone_labels, valid, logits.shape[1], logits.shape[-1]
            )
            ce = F.cross_entropy(logits, lab, reduction="none")
            phone = (ce * mask).sum() / mask.sum().clamp(min=1.0)
        else:
            phone = torch.zeros((), device=logits.device)

        total = (self.w_adv * adv + self.w_fm * fm) * self.gain + self.w_phone * phone \
            + self.w_pitch * (pitch_adv + pitch_fm) * self.gain
        self._set_d_grad(True)
        return {
            "total": total,
            "adv": adv.detach(),
            "fm": fm.detach(),
            "phone": phone.detach(),
            "pitch": pitch_adv.detach(),
        }


def cosine_lr(step, peak_lr, min_lr, warmup_step, total_step):
    """线性 warmup + 余弦衰减到 min_lr。

    step 由训练循环传入而不是优化器内部自增, 这样从 checkpoint 恢复后落在
    和不中断训练完全相同的 lr 上 (增量式 lr 在 resume 时最容易悄悄错位)。
    """
    warmup_step = max(0, int(warmup_step))
    if warmup_step > 0 and step < warmup_step:
        return peak_lr * (step + 1) / warmup_step
    span = max(1, int(total_step) - warmup_step)
    progress = min(1.0, max(0.0, (step - warmup_step) / span))
    return min_lr + 0.5 * (peak_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


class CosineLRAdam:
    """Adam 封装。lr = warmup(线性) -> 余弦衰减, 由外部 step 驱动。"""

    def __init__(self, model, train_config):
        opt = train_config["optimizer"]
        self.lr = float(opt["lr"])
        self.min_lr = float(opt.get("min_lr", self.lr * 0.01))
        self.warmup_step = int(opt.get("warmup_step", 0))
        self.total_step = int(train_config["step"]["total_step"])
        self.current_lr = self.lr
        self._last_step = 0
        self._optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.lr,
            betas=opt["betas"],
            eps=opt["eps"],
            weight_decay=opt["weight_decay"],
        )
        self.set_step(0)

    def set_step(self, step):
        self._last_step = int(step)
        self.current_lr = cosine_lr(
            self._last_step, self.lr, self.min_lr, self.warmup_step, self.total_step
        )
        for param_group in self._optimizer.param_groups:
            param_group["lr"] = self.current_lr
        return self.current_lr

    def step(self, step=None):
        if step is not None:
            self.set_step(step)
        self._optimizer.step()

    def zero_grad(self):
        self._optimizer.zero_grad()

    def load_state_dict(self, state):
        # 不在这里写 lr: 下一个 set_step 会按全局 step 重算, 覆盖掉
        # state_dict 里保存的旧 lr。
        self._optimizer.load_state_dict(state)
        self.set_step(self._last_step)


def merge_config(base, extra):
    merged = dict(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_config(merged[key], value)
        else:
            merged[key] = value
    return merged


def get_checkpoint_config(train_config, config_path=None):
    checkpoint_config = train_config.get("checkpoint") or {}
    missing = [
        key for key in ("best_file",) if not checkpoint_config.get(key)
    ]
    if missing:
        where = " in {}".format(config_path) if config_path else ""
        raise SystemExit(
            "Missing required config{}: {}".format(
                where,
                ", ".join("checkpoint." + key for key in missing),
            )
        )
    return checkpoint_config


def get_last_file(train_config, config_path=None):
    best_file = get_checkpoint_config(train_config, config_path)["best_file"]
    stem, ext = os.path.splitext(best_file)
    return "{}(last){}".format(stem, ext)


def get_model(args, configs, device):
    (preprocess_config, model_config, train_config) = configs

    model = FastSinger(preprocess_config, model_config).to(device)
    ckpt_path = os.path.join(
        train_config["path"]["ckpt_path"], get_last_file(train_config)
    )

    ckpt = None
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        args.restore_step = int(ckpt.get("step", 0))
        print(
            "Resume from {} (step {})".format(
                os.path.basename(ckpt_path), args.restore_step
            )
        )

    optim = CosineLRAdam(model, train_config)
    if ckpt is not None:
        optim.load_state_dict(ckpt["optimizer"])

    model.train()
    return model, optim


def get_param_num(model):
    num_param = sum(param.numel() for param in model.parameters())
    return num_param


def get_vocoder(device, train_config):
    vocoder_config = train_config.get("vocoder") or {}
    missing = [key for key in ("path", "file") if not vocoder_config.get(key)]
    if missing:
        raise SystemExit(
            "Missing required config in {}: {}".format(
                TRAIN_CONFIG_PATH,
                ", ".join("vocoder." + key for key in missing),
            )
        )
    vocoder_dir = vocoder_config["path"]
    vocoder_arch_path = os.path.join(
        vocoder_dir, vocoder_config.get("config", "config.json")
    )
    vocoder_ckpt_path = os.path.join(vocoder_dir, vocoder_config["file"])
    if not os.path.isdir(vocoder_dir):
        raise SystemExit("Vocoder directory not found: {}".format(vocoder_dir))
    if not os.path.isfile(vocoder_arch_path):
        raise SystemExit("Vocoder config not found: {}".format(vocoder_arch_path))
    if not os.path.isfile(vocoder_ckpt_path):
        raise SystemExit("Vocoder checkpoint not found: {}".format(vocoder_ckpt_path))

    with open(vocoder_arch_path, "r") as f:
        vocoder_arch = AttrDict(json.load(f))
    vocoder = Vocoder(vocoder_arch)
    ckpt = torch.load(
        vocoder_ckpt_path,
        map_location=device,
        weights_only=False,
    )
    vocoder.load_state_dict(ckpt["generator"])
    vocoder.eval()
    vocoder.remove_weight_norm()
    vocoder.to(device)

    return vocoder


def vocoder_infer(mels, vocoder, preprocess_config, lengths=None):
    with torch.no_grad():
        wavs = vocoder(mels).squeeze(1)

    wavs = (
        wavs.cpu().numpy()
        * preprocess_config["audio"]["max_wav_value"]
    ).astype("int16")
    wavs = [wav for wav in wavs]

    for i in range(len(mels)):
        if lengths is not None:
            wavs[i] = wavs[i][: lengths[i]]

    return wavs


def to_device(data, device):
    (
        ids,
        raw_texts,
        texts,
        src_lens,
        max_src_len,
        mels,
        mel_lens,
        max_mel_len,
        pitches,
        energies,
        durations,
    ) = data

    texts = torch.from_numpy(texts).long().to(device)
    src_lens = torch.from_numpy(src_lens).to(device)
    mels = torch.from_numpy(mels).float().to(device)
    mel_lens = torch.from_numpy(mel_lens).to(device)
    pitches = torch.from_numpy(pitches).float().to(device)
    energies = torch.from_numpy(energies).float().to(device)
    durations = torch.from_numpy(durations).long().to(device)

    return (
        ids,
        raw_texts,
        texts,
        src_lens,
        max_src_len,
        mels,
        mel_lens,
        max_mel_len,
        pitches,
        energies,
        durations,
    )


def log(
    logger, step=None, losses=None, fig=None, audio=None, sampling_rate=22050, tag=""
):
    if losses is not None:
        logger.add_scalar("Loss/total_loss", losses[0], step)
        logger.add_scalar("Loss/mel_loss", losses[1], step)
        logger.add_scalar("Loss/mel_postnet_loss", losses[2], step)
        logger.add_scalar("Loss/energy_loss", losses[3], step)

    if fig is not None:
        logger.add_figure(tag, fig)

    if audio is not None:
        logger.add_audio(
            tag,
            audio / max(abs(audio)),
            sample_rate=sampling_rate,
        )


def get_mask_from_lengths(lengths, max_len=None):
    batch_size = lengths.shape[0]
    if max_len is None:
        max_len = torch.max(lengths).item()

    ids = torch.arange(0, max_len, device=lengths.device).unsqueeze(0).expand(batch_size, -1)
    mask = ids >= lengths.unsqueeze(1).expand(-1, max_len)

    return mask


def expand(values, durations):
    out = list()
    for value, d in zip(values, durations):
        out += [value] * max(0, int(d))
    return np.array(out)


def load_stats(preprocess_config):
    with open(
        os.path.join(preprocess_config["path"]["preprocessed_path"], "stats.json")
    ) as f:
        return json.load(f)


def synth_one_sample(targets, predictions, vocoder, model_config, preprocess_config):
    basename = targets[0][0]
    src_len = predictions[6][0].item()
    mel_len = predictions[7][0].item()
    mel_target = targets[5][0, :mel_len].detach().transpose(0, 1)
    mel_prediction = predictions[1][0, :mel_len].detach().transpose(0, 1)
    duration = targets[10][0, :src_len].detach().cpu().numpy()
    f0 = targets[8][0, :mel_len].detach().cpu().numpy()
    energy = targets[9][0, :src_len].detach().cpu().numpy()
    energy = expand(energy, duration)

    stats = load_stats(preprocess_config)

    fig = plot_mel(
        [
            (mel_prediction.cpu().numpy(), f0, energy),
            (mel_target.cpu().numpy(), f0, energy),
        ],
        stats,
        ["Synthesized Spectrogram", "Ground-Truth Spectrogram"],
    )

    if vocoder is not None:
        wav_reconstruction = vocoder_infer(
            mel_target.unsqueeze(0),
            vocoder,
            preprocess_config,
        )[0]
        wav_prediction = vocoder_infer(
            mel_prediction.unsqueeze(0),
            vocoder,
            preprocess_config,
        )[0]
    else:
        wav_reconstruction = wav_prediction = None

    return fig, wav_reconstruction, wav_prediction, basename


def synth_samples(targets, predictions, vocoder, model_config, preprocess_config, path):
    basenames = targets[0]
    stats = load_stats(preprocess_config)
    for i in range(len(predictions[0])):
        basename = basenames[i]
        src_len = predictions[6][i].item()
        mel_len = predictions[7][i].item()
        mel_prediction = predictions[1][i, :mel_len].detach().transpose(0, 1)
        duration = predictions[3][i, :src_len].detach().cpu().numpy()
        f0 = targets[8][i, :mel_len].detach().cpu().numpy()
        energy = predictions[2][i, :src_len].detach().cpu().numpy()
        energy = expand(energy, duration)

        fig = plot_mel(
            [
                (mel_prediction.cpu().numpy(), f0, energy),
            ],
            stats,
            ["Synthesized Spectrogram"],
        )
        plt.savefig(os.path.join(path, "{}.png".format(basename)))
        plt.close()

    mel_predictions = predictions[1].transpose(1, 2)
    lengths = predictions[7] * preprocess_config["audio"]["hop_length"]
    wav_predictions = vocoder_infer(
        mel_predictions, vocoder, preprocess_config, lengths=lengths
    )

    sampling_rate = preprocess_config["audio"]["sampling_rate"]
    for wav, basename in zip(wav_predictions, basenames):
        wavfile.write(os.path.join(path, "{}.wav".format(basename)), sampling_rate, wav)


def plot_mel(data, stats, titles):
    fig, axes = plt.subplots(len(data), 1, squeeze=False)
    if titles is None:
        titles = [None for i in range(len(data))]
    pitch_min, pitch_max = stats["pitch"][:2]
    energy_min, energy_max = stats["energy"][:2]

    def add_axis(fig, old_ax):
        ax = fig.add_axes(old_ax.get_position(), anchor="W")
        ax.set_facecolor("None")
        return ax

    for i in range(len(data)):
        mel, f0, energy = data[i]
        axes[i][0].imshow(mel, origin="lower")
        axes[i][0].set_aspect(2.5, adjustable="box")
        axes[i][0].set_ylim(0, mel.shape[0])
        axes[i][0].set_title(titles[i], fontsize="medium")
        axes[i][0].tick_params(labelsize="x-small", left=False, labelleft=False)
        axes[i][0].set_anchor("W")

        ax1 = add_axis(fig, axes[i][0])
        ax1.plot(f0, color="tomato")
        ax1.set_xlim(0, mel.shape[1])
        ax1.set_ylim(0, pitch_max)
        ax1.set_ylabel("F0", color="tomato")
        ax1.tick_params(
            labelsize="x-small", colors="tomato", bottom=False, labelbottom=False
        )

        ax2 = add_axis(fig, axes[i][0])
        ax2.plot(energy, color="darkviolet")
        ax2.set_xlim(0, mel.shape[1])
        ax2.set_ylim(energy_min, energy_max)
        ax2.set_ylabel("Energy", color="darkviolet")
        ax2.yaxis.set_label_position("right")
        ax2.tick_params(
            labelsize="x-small",
            colors="darkviolet",
            bottom=False,
            labelbottom=False,
            left=False,
            labelleft=False,
            right=True,
            labelright=True,
        )

    return fig


def pad_1D(inputs, PAD=0):
    def pad_data(x, length, PAD):
        x_padded = np.pad(
            x, (0, length - x.shape[0]), mode="constant", constant_values=PAD
        )
        return x_padded

    max_len = max((len(x) for x in inputs))
    padded = np.stack([pad_data(x, max_len, PAD) for x in inputs])

    return padded


def pad_2D(inputs, maxlen=None):
    def pad(x, max_len):
        PAD = 0
        if np.shape(x)[0] > max_len:
            raise ValueError("not max_len")

        s = np.shape(x)[1]
        x_padded = np.pad(
            x, (0, max_len - np.shape(x)[0]), mode="constant", constant_values=PAD
        )
        return x_padded[:, :s]

    if maxlen:
        output = np.stack([pad(x, maxlen) for x in inputs])
    else:
        max_len = max(np.shape(x)[0] for x in inputs)
        output = np.stack([pad(x, max_len) for x in inputs])

    return output


def pad(input_ele, mel_max_length=None):
    if mel_max_length:
        max_len = mel_max_length
    else:
        max_len = max([input_ele[i].size(0) for i in range(len(input_ele))])

    out_list = list()
    for i, batch in enumerate(input_ele):
        if len(batch.shape) == 1:
            one_batch_padded = F.pad(
                batch, (0, max_len - batch.size(0)), "constant", 0.0
            )
        elif len(batch.shape) == 2:
            one_batch_padded = F.pad(
                batch, (0, 0, 0, max_len - batch.size(0)), "constant", 0.0
            )
        out_list.append(one_batch_padded)
    out_padded = torch.stack(out_list)
    return out_padded


class Dataset(Dataset):
    def __init__(
        self, filename, preprocess_config, train_config, sort=False, drop_last=False
    ):
        self.preprocessed_path = preprocess_config["path"]["preprocessed_path"]
        self.batch_size = train_config["optimizer"]["batch_size"]

        self.basename, self.text, self.raw_text = self.process_meta(filename)
        self.sort = sort
        self.drop_last = drop_last

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        basename = self.basename[idx]
        raw_text = self.raw_text[idx]
        phone = np.array(phone_to_sequence(self.text[idx]))
        mel_path = os.path.join(self.preprocessed_path, "mel", "{}-mel.npy".format(basename))
        mel = np.load(mel_path)
        pitch_path = os.path.join(self.preprocessed_path, "pitch", "{}-pitch.npy".format(basename))
        pitch = np.load(pitch_path)
        energy_path = os.path.join(self.preprocessed_path, "energy", "{}-energy.npy".format(basename))
        energy = np.load(energy_path)
        duration_path = os.path.join(self.preprocessed_path, "duration", "{}-duration.npy".format(basename))
        duration = np.load(duration_path)

        sample = {
            "id": basename,
            "text": phone,
            "raw_text": raw_text,
            "mel": mel,
            "pitch": pitch,
            "energy": energy,
            "duration": duration,
        }

        return sample

    def process_meta(self, filename):
        with open(
            os.path.join(self.preprocessed_path, filename), "r", encoding="utf-8"
        ) as f:
            name = []
            text = []
            raw_text = []
            for line in f.readlines():
                n, t, r = line.strip("\n").split("|")
                name.append(n)
                text.append(t)
                raw_text.append(r)
            return name, text, raw_text

    def reprocess(self, data, idxs):
        ids = [data[idx]["id"] for idx in idxs]
        texts = [data[idx]["text"] for idx in idxs]
        raw_texts = [data[idx]["raw_text"] for idx in idxs]
        mels = [data[idx]["mel"] for idx in idxs]
        pitches = [data[idx]["pitch"] for idx in idxs]
        energies = [data[idx]["energy"] for idx in idxs]
        durations = [data[idx]["duration"] for idx in idxs]

        text_lens = np.array([text.shape[0] for text in texts])
        mel_lens = np.array([mel.shape[0] for mel in mels])

        texts = pad_1D(texts)
        mels = pad_2D(mels)
        pitches = pad_1D(pitches)
        energies = pad_1D(energies)
        durations = pad_1D(durations)

        return (
            ids,
            raw_texts,
            texts,
            text_lens,
            max(text_lens),
            mels,
            mel_lens,
            max(mel_lens),
            pitches,
            energies,
            durations,
        )

    def collate_fn(self, data):
        data_size = len(data)

        if self.sort:
            # Bucket by mel frame length, not by phone count. Phone count is
            # nearly constant for single-syllable corpora (1 or 2), so sorting
            # by it gives zero padding benefit, while mel length varies widely
            # and drives almost all of the padding waste.
            len_arr = np.array([d["mel"].shape[0] for d in data])
            idx_arr = np.argsort(-len_arr)
        else:
            idx_arr = np.arange(data_size)

        tail = idx_arr[len(idx_arr) - (len(idx_arr) % self.batch_size) :]
        idx_arr = idx_arr[: len(idx_arr) - (len(idx_arr) % self.batch_size)]
        idx_arr = idx_arr.reshape((-1, self.batch_size)).tolist()
        if not self.drop_last and len(tail) > 0:
            idx_arr += [tail.tolist()]

        output = list()
        for idx in idx_arr:
            output.append(self.reprocess(data, idx))

        return output


# RMVPE 基频提取器 (https://arxiv.org/abs/2306.15412)
# 输入 16 kHz 音频, 输出 100 帧/秒 的 F0 (Hz)
SAMPLE_RATE = 16000
HOP_LENGTH = 160
N_MELS = 128
N_CLASS = 360
MIN_AUDIO_LEN = 1024


class BiGRU(nn.Module):
    def __init__(self, input_features, hidden_features, num_layers):
        super(BiGRU, self).__init__()
        self.gru = nn.GRU(
            input_features,
            hidden_features,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, x):
        return self.gru(x)[0]


class ConvBlockRes(nn.Module):
    def __init__(self, in_channels, out_channels, momentum=0.01):
        super(ConvBlockRes, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(),
        )
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, (1, 1))

    def forward(self, x):
        if not hasattr(self, "shortcut"):
            return self.conv(x) + x
        return self.conv(x) + self.shortcut(x)


class ResEncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, n_blocks, momentum=0.01):
        super(ResEncoderBlock, self).__init__()
        self.conv = nn.ModuleList()
        for _ in range(n_blocks):
            self.conv.append(ConvBlockRes(in_channels, out_channels, momentum))
            in_channels = out_channels
        self.pool = None
        if kernel_size is not None:
            self.pool = nn.AvgPool2d(kernel_size=kernel_size)

    def forward(self, x):
        for conv in self.conv:
            x = conv(x)
        if self.pool is not None:
            return x, self.pool(x)
        return x


class RMVEncoder(nn.Module):
    def __init__(
        self,
        in_channels,
        in_size,
        n_encoders,
        kernel_size,
        n_blocks,
        out_channels=16,
        momentum=0.01,
    ):
        super(RMVEncoder, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels, momentum=momentum)
        self.layers = nn.ModuleList()
        for _ in range(n_encoders):
            self.layers.append(
                ResEncoderBlock(
                    in_channels, out_channels, kernel_size, n_blocks, momentum=momentum
                )
            )
            in_channels = out_channels
            out_channels *= 2
            in_size //= 2
        self.out_size = in_size
        self.out_channel = out_channels

    def forward(self, x):
        concat_tensors = []
        x = self.bn(x)
        for layer in self.layers:
            t, x = layer(x)
            concat_tensors.append(t)
        return x, concat_tensors


class Intermediate(nn.Module):
    def __init__(self, in_channels, out_channels, n_inters, n_blocks, momentum=0.01):
        super(Intermediate, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(
            ResEncoderBlock(in_channels, out_channels, None, n_blocks, momentum)
        )
        for _ in range(n_inters - 1):
            self.layers.append(
                ResEncoderBlock(out_channels, out_channels, None, n_blocks, momentum)
            )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ResDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, n_blocks, momentum=0.01):
        super(ResDecoderBlock, self).__init__()
        out_padding = (0, 1) if stride == (1, 2) else (1, 1)
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(3, 3),
                stride=stride,
                padding=(1, 1),
                output_padding=out_padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(),
        )
        self.conv2 = nn.ModuleList()
        self.conv2.append(ConvBlockRes(out_channels * 2, out_channels, momentum))
        for _ in range(n_blocks - 1):
            self.conv2.append(ConvBlockRes(out_channels, out_channels, momentum))

    def forward(self, x, concat_tensor):
        x = self.conv1(x)
        x = torch.cat((x, concat_tensor), dim=1)
        for conv2 in self.conv2:
            x = conv2(x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_channels, n_decoders, stride, n_blocks, momentum=0.01):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList()
        for _ in range(n_decoders):
            out_channels = in_channels // 2
            self.layers.append(
                ResDecoderBlock(in_channels, out_channels, stride, n_blocks, momentum)
            )
            in_channels = out_channels

    def forward(self, x, concat_tensors):
        for i, layer in enumerate(self.layers):
            x = layer(x, concat_tensors[-1 - i])
        return x


class DeepUnet(nn.Module):
    def __init__(
        self,
        kernel_size,
        n_blocks,
        en_de_layers=5,
        inter_layers=4,
        in_channels=1,
        en_out_channels=16,
    ):
        super(DeepUnet, self).__init__()
        self.encoder = RMVEncoder(
            in_channels, 128, en_de_layers, kernel_size, n_blocks, en_out_channels
        )
        self.intermediate = Intermediate(
            self.encoder.out_channel // 2,
            self.encoder.out_channel,
            inter_layers,
            n_blocks,
        )
        self.decoder = Decoder(
            self.encoder.out_channel, en_de_layers, kernel_size, n_blocks
        )

    def forward(self, x):
        x, concat_tensors = self.encoder(x)
        x = self.intermediate(x)
        x = self.decoder(x, concat_tensors)
        return x


class E2E(nn.Module):
    def __init__(
        self,
        n_blocks,
        n_gru,
        kernel_size,
        en_de_layers=5,
        inter_layers=4,
        in_channels=1,
        en_out_channels=16,
    ):
        super(E2E, self).__init__()
        self.unet = DeepUnet(
            kernel_size,
            n_blocks,
            en_de_layers,
            inter_layers,
            in_channels,
            en_out_channels,
        )
        self.cnn = nn.Conv2d(en_out_channels, 3, (3, 3), padding=(1, 1))
        self.fc = nn.Sequential(
            BiGRU(3 * 128, 256, n_gru),
            nn.Linear(512, N_CLASS),
            nn.Dropout(0.25),
            nn.Sigmoid(),
        )

    def forward(self, mel):
        mel = mel.transpose(-1, -2).unsqueeze(1)
        x = self.cnn(self.unet(mel)).transpose(1, 2).flatten(-2)
        x = self.fc(x)
        return x


class MelSpectrogram(nn.Module):
    def __init__(
        self,
        n_mel_channels=N_MELS,
        sampling_rate=SAMPLE_RATE,
        win_length=1024,
        hop_length=HOP_LENGTH,
        n_fft=None,
        mel_fmin=30,
        mel_fmax=8000,
        clamp=1e-5,
    ):
        super().__init__()
        n_fft = win_length if n_fft is None else n_fft
        mel_basis = librosa_mel_fn(
            sr=sampling_rate,
            n_fft=n_fft,
            n_mels=n_mel_channels,
            fmin=mel_fmin,
            fmax=mel_fmax,
            htk=True,
        )
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer("mel_basis", mel_basis)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.clamp = clamp
        self.hann_window = {}

    def forward(self, audio, center=True):
        if self.win_length not in self.hann_window:
            self.hann_window[self.win_length] = torch.hann_window(
                self.win_length, device=audio.device
            )
        fft = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.hann_window[self.win_length],
            center=center,
            return_complex=True,
        )
        magnitude = torch.sqrt(fft.real.pow(2) + fft.imag.pow(2))
        mel_output = torch.matmul(self.mel_basis, magnitude)
        return torch.log(torch.clamp(mel_output, min=self.clamp))


class RMVPE:
    """Robust MVPE pitch estimator (https://arxiv.org/abs/2306.15412).

    Expects audio sampled at 16 kHz and returns F0 in Hz at 100 frames/sec.
    """

    def __init__(self, model_path, device=None, is_half=False):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.is_half = bool(is_half) and self.device.type == "cuda"

        self.model = E2E(4, 1, (2, 2))
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and "model" in ckpt:
            ckpt = ckpt["model"]
        self.model.load_state_dict(ckpt)
        self.model.eval()
        if self.is_half:
            self.model = self.model.half()
        else:
            self.model = self.model.float()
        self.model.to(self.device)

        self.mel_extractor = MelSpectrogram().to(self.device)
        cents_mapping = 20 * np.arange(N_CLASS) + 1997.3794084376191
        self.cents_mapping = np.pad(cents_mapping, (4, 4))

    @torch.no_grad()
    def extract_mel(self, audio, center=True):
        if not torch.is_tensor(audio):
            audio = torch.from_numpy(np.asarray(audio, dtype=np.float32))
        audio = audio.float().to(self.device)
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        if audio.shape[-1] < MIN_AUDIO_LEN:
            audio = F.pad(audio, (0, MIN_AUDIO_LEN - audio.shape[-1]))
        return self.mel_extractor(audio, center=center)

    @torch.no_grad()
    def mel2hidden(self, mel):
        n_frames = mel.shape[-1]
        n_pad = 32 * ((n_frames - 1) // 32 + 1) - n_frames
        if n_pad > 0:
            mel = F.pad(mel, (0, n_pad), mode="constant")
        mel = mel.half() if self.is_half else mel.float()
        hidden = self.model(mel)
        return hidden[:, :n_frames]

    def decode(self, hidden, thred=0.03):
        cents_pred = self.to_local_average_cents(hidden, thred=thred)
        f0 = 10 * (2 ** (cents_pred / 1200))
        f0[f0 == 10] = 0
        return f0

    def infer_from_audio(self, audio, thred=0.03):
        mel = self.extract_mel(audio, center=True)
        hidden = self.mel2hidden(mel)
        hidden = hidden.squeeze(0).float().cpu().numpy()
        return self.decode(hidden, thred=thred)

    def to_local_average_cents(self, salience, thred=0.03):
        n_frames = salience.shape[0]
        center = np.argmax(salience, axis=1)
        salience = np.pad(salience, ((0, 0), (4, 4)))
        center = center + 4
        idx = center[:, None] - 4 + np.arange(9)[None, :]
        rows = np.arange(n_frames)[:, None]
        window = salience[rows, idx]
        mapping = self.cents_mapping[idx]
        product_sum = np.sum(window * mapping, axis=1)
        weight_sum = np.sum(window, axis=1)
        cents = product_sum / np.maximum(weight_sum, 1e-12)
        cents[np.max(salience, axis=1) <= thred] = 0
        return cents


def load_rmvpe(model_path, device=None):
    if not os.path.exists(model_path):
        raise SystemExit(
            "RMVPE weights not found: {}\n"
            "Download rmvpe.pt and put it there.".format(model_path)
        )
    return RMVPE(model_path, device=device, is_half=False)


class Preprocessor:
    def __init__(self, config):
        self.config = config
        self.in_dir = config["path"]["corpus_path"]
        self.out_dir = config["path"]["preprocessed_path"]
        self.val_size = config["preprocessing"]["val_size"]
        self.sampling_rate = config["audio"]["sampling_rate"]
        self.hop_length = config["audio"]["hop_length"]
        self.energy_normalization = config["preprocessing"]["energy"]["normalization"]

        self.STFT = LogMelSTFT(
            config["preprocessing"]["stft"]["filter_length"],
            config["audio"]["hop_length"],
            config["preprocessing"]["stft"]["win_length"],
            config["audio"]["n_mel_channels"],
            config["audio"]["sampling_rate"],
            config["preprocessing"]["mel"]["mel_fmin"],
            config["preprocessing"]["mel"]["mel_fmax"],
        )

        self.rmvpe = load_rmvpe(
            os.path.join(config["path"]["ckpt_path"], "rmvpe.pt"), device=device
        )

    def extract_f0(self, wav, total):
        wav_16k = librosa.resample(
            wav, orig_sr=self.sampling_rate, target_sr=16000
        )
        f0 = self.rmvpe.infer_from_audio(wav_16k)
        if f0.size == 0:
            return np.zeros(total, dtype=np.float32)
        source = np.round(
            np.arange(total) * self.hop_length / self.sampling_rate * 100.0
        ).astype(np.int64)
        np.clip(source, 0, f0.size - 1, out=source)
        return f0[source].astype(np.float32)

    def build_from_path(self):
        os.makedirs((os.path.join(self.out_dir, "mel")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "pitch")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "energy")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "duration")), exist_ok=True)

        print("Processing Data ...")
        out = list()
        n_frames = 0
        energy_scaler = StandardScaler()
        utt_energies = []
        pitch_values = []
        phone_dur_sum = defaultdict(float)
        phone_dur_cnt = defaultdict(int)
        phone_voiced_frames = defaultdict(int)
        phone_frames = defaultdict(int)
        pinyin2phones = {}

        wav_files = sorted(f for f in os.listdir(self.in_dir) if f.endswith(".wav"))
        for wav_name in tqdm(wav_files):
            stem = wav_name[:-4]
            match = re.match(r"^(.+)\(\1\)\((\d+)\)(?:\(([\d.]+)\))?$", stem)
            if match is None:
                print("Skip (bad name): {}".format(stem))
                continue
            pinyin = match.group(1)
            utt_phones = pinyin_to_phones(pinyin)

            ret = self.process_utterance(stem, utt_phones)
            if ret is None:
                continue
            info, f0, energy, n, duration, utt_phones = ret
            out.append(info)
            n_frames += n

            voiced = self.remove_outlier(f0[f0 > 0])
            if len(voiced) > 0:
                pitch_values.extend(voiced.tolist())
            utt_energies.append((stem, energy))
            norm_energy = self.remove_outlier(energy)
            if len(norm_energy) > 0:
                energy_scaler.partial_fit(norm_energy.reshape((-1, 1)))

            pos = 0
            for p, d in zip(utt_phones, duration):
                phone_dur_sum[p] += float(d)
                phone_dur_cnt[p] += 1
                seg = f0[pos : pos + d]
                phone_voiced_frames[p] += int(np.sum(seg > 0))
                phone_frames[p] += int(d)
                pos += d
            pinyin2phones.setdefault(pinyin, list(utt_phones))

        if len(pitch_values) == 0:
            raise RuntimeError("No valid utterances found in {}".format(self.in_dir))

        print("Computing statistic quantities ...")
        if self.energy_normalization:
            energy_mean = energy_scaler.mean_[0]
            energy_std = energy_scaler.scale_[0]
        else:
            energy_mean = 0
            energy_std = 1

        pitch_min = float(min(pitch_values))
        pitch_max = float(max(pitch_values))
        energy_min = np.finfo(np.float64).max
        energy_max = np.finfo(np.float64).min
        for stem, energy in utt_energies:
            values = (energy - energy_mean) / energy_std
            np.save(
                os.path.join(self.out_dir, "energy", "{}-energy.npy".format(stem)),
                values,
            )
            if len(values) > 0:
                energy_min = min(energy_min, float(values.min()))
                energy_max = max(energy_max, float(values.max()))

        phone_mean_duration = {
            p: phone_dur_sum[p] / phone_dur_cnt[p] for p in phone_dur_cnt
        }
        phone_voiced_ratio = {
            p: phone_voiced_frames[p] / phone_frames[p] for p in phone_frames
        }

        with open(os.path.join(self.out_dir, "stats.json"), "w") as f:
            stats = {
                "pitch": [pitch_min, pitch_max],
                "energy": [
                    float(energy_min),
                    float(energy_max),
                    float(energy_mean),
                    float(energy_std),
                ],
                "phone_mean_duration": phone_mean_duration,
                "phone_voiced_ratio": phone_voiced_ratio,
            }
            f.write(json.dumps(stats))

        with open(os.path.join(self.out_dir, "pinyin2phones.json"), "w") as f:
            f.write(json.dumps(pinyin2phones, ensure_ascii=False, indent=0))

        print(
            "Total time: {} hours".format(
                n_frames * self.hop_length / self.sampling_rate / 3600
            )
        )
        print("Utterances: {}".format(len(out)))
        print("Pinyins: {}".format(len(pinyin2phones)))

        random.seed(1234)
        random.shuffle(out)
        out = [r for r in out if r is not None]

        with open(os.path.join(self.out_dir, "train.txt"), "w", encoding="utf-8") as f:
            for m in out[self.val_size :]:
                f.write(m + "\n")
        with open(os.path.join(self.out_dir, "val.txt"), "w", encoding="utf-8") as f:
            for m in out[: self.val_size]:
                f.write(m + "\n")

        return out

    def process_utterance(self, stem, utt_phones):
        wav_path = os.path.join(self.in_dir, "{}.wav".format(stem))

        match = re.match(r"^(.+)\(\1\)\((\d+)\)(?:\(([\d.]+)\))?$", stem)
        if match is None:
            print("Skip (bad name): {}".format(stem))
            return None
        pinyin = match.group(1)
        text = "{" + " ".join(utt_phones) + "}"

        wav, _ = librosa.load(wav_path, sr=self.sampling_rate)
        mel_spectrogram, energy = get_mel_from_wav(wav, self.STFT)
        total = int(mel_spectrogram.shape[1])
        if total <= 0:
            print("Skip (empty audio): {}".format(stem))
            return None

        if len(utt_phones) == 1:
            duration = np.array([total], dtype=np.int64)
        elif total < 2:
            print("Skip (audio too short for two phones): {}".format(stem))
            return None
        else:
            boundary = find_initial_boundary(mel_spectrogram)
            boundary = int(np.clip(boundary, 1, total - 1))
            duration = np.array([boundary, total - boundary], dtype=np.int64)

        if np.any(duration <= 0):
            print("Skip (degenerate duration): {}".format(stem))
            return None
        energy = energy[:total]

        f0 = self.extract_f0(wav, total)
        pos = 0
        for p, d in zip(utt_phones, duration):
            if p in VOICELESS_INITIALS:
                f0[pos : pos + d] = 0.0
            pos += d
        if np.sum(f0 != 0) <= 1:
            print("Skip (no voiced frames): {}".format(stem))
            return None

        pos = 0
        for i, d in enumerate(duration):
            if d > 0:
                energy[i] = np.mean(energy[pos : pos + d])
            else:
                energy[i] = 0
            pos += d
        energy = energy[: len(utt_phones)]

        dur_filename = "{}-duration.npy".format(stem)
        np.save(os.path.join(self.out_dir, "duration", dur_filename), duration)

        pitch_filename = "{}-pitch.npy".format(stem)
        np.save(os.path.join(self.out_dir, "pitch", pitch_filename), f0.astype(np.float32))

        mel_filename = "{}-mel.npy".format(stem)
        np.save(
            os.path.join(self.out_dir, "mel", mel_filename),
            mel_spectrogram.T,
        )

        return (
            "|".join([stem, text, pinyin]),
            self.remove_outlier(f0[f0 > 0]),
            energy,
            total,
            duration,
            utt_phones,
        )

    def remove_outlier(self, values):
        values = np.array(values)
        if values.size < 2:
            return values
        p25 = np.percentile(values, 25)
        p75 = np.percentile(values, 75)
        lower = p25 - 1.5 * (p75 - p25)
        upper = p75 + 1.5 * (p75 - p25)
        normal_indices = np.logical_and(values > lower, values < upper)

        return values[normal_indices]


def evaluate(model, step, configs, logger=None, vocoder=None):
    preprocess_config, model_config, train_config = configs

    dataset = Dataset(
        "val.txt", preprocess_config, train_config, sort=False, drop_last=False
    )
    batch_size = train_config["optimizer"]["batch_size"]
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dataset.collate_fn,
        num_workers=2,
    )

    Loss = FastSingerLoss(preprocess_config, model_config).to(device)

    loss_sums = [0 for _ in range(4)]
    for batchs in loader:
        for batch in batchs:
            batch = to_device(batch, device)
            with torch.no_grad():
                output = model(*(batch[2:]))
                losses = Loss(batch, output)

                for i in range(len(losses)):
                    loss_sums[i] += losses[i].item() * len(batch[0])

    loss_means = [loss_sum / len(dataset) for loss_sum in loss_sums]

    message = "Validation Step {}, Total Loss: {:.4f}, Mel Loss: {:.4f}, Mel PostNet Loss: {:.4f}, Energy Loss: {:.4f}".format(
        *([step] + [l for l in loss_means])
    )

    if logger is not None:
        fig, wav_reconstruction, wav_prediction, tag = synth_one_sample(
            batch,
            output,
            vocoder,
            model_config,
            preprocess_config,
        )

        log(logger, step, losses=loss_means)
        log(
            logger,
            fig=fig,
            tag="Validation/step_{}_{}".format(step, tag),
        )
        sampling_rate = preprocess_config["audio"]["sampling_rate"]
        log(
            logger,
            audio=wav_reconstruction,
            sampling_rate=sampling_rate,
            tag="Validation/step_{}_{}_reconstructed".format(step, tag),
        )
        log(
            logger,
            audio=wav_prediction,
            sampling_rate=sampling_rate,
            tag="Validation/step_{}_{}_synthesized".format(step, tag),
        )

    return message, loss_means


def run_train(args, configs):
    print("Prepare training ...")

    preprocess_config, model_config, train_config = configs
    checkpoint_config = get_checkpoint_config(train_config)
    ckpt_path = train_config["path"]["ckpt_path"]

    for p in train_config["path"].values():
        os.makedirs(p, exist_ok=True)
    train_log_path = os.path.join(train_config["path"]["log_path"], "train")
    val_log_path = os.path.join(train_config["path"]["log_path"], "val")
    os.makedirs(train_log_path, exist_ok=True)
    os.makedirs(val_log_path, exist_ok=True)
    train_logger = SummaryWriter(train_log_path)
    val_logger = SummaryWriter(val_log_path)

    dataset = Dataset(
        "train.txt", preprocess_config, train_config, sort=True, drop_last=True
    )
    batch_size = train_config["optimizer"]["batch_size"]
    group_size = 4
    assert batch_size * group_size < len(dataset)
    loader = DataLoader(
        dataset,
        batch_size=batch_size * group_size,
        shuffle=True,
        collate_fn=dataset.collate_fn,
        num_workers=4,
        persistent_workers=True,
        prefetch_factor=4,
    )

    model, optimizer = get_model(args, configs, device)
    model = nn.DataParallel(model)
    num_param = get_param_num(model)
    Loss = FastSingerLoss(preprocess_config, model_config).to(device)
    Adv = Adversarial(preprocess_config, model_config, train_config).to(device)
    print(
        "LR schedule: {:.2e} -> {:.2e} (warmup {} steps, cosine over {} steps)".format(
            optimizer.lr, optimizer.min_lr, optimizer.warmup_step, optimizer.total_step
        )
    )
    if Adv.enabled:
        print(
            "Adversarial: adv={} fm={} phone={} pitch={} (from step {})".format(
                Adv.w_adv, Adv.w_fm, Adv.w_phone, Adv.w_pitch, Adv.start_step
            )
        )
    print("Number of FastSinger Parameters:", num_param)

    vocoder = get_vocoder(device, train_config)

    resume_step = getattr(args, "restore_step", 0)
    step = resume_step + 1
    epoch = 1
    grad_acc_step = train_config["optimizer"]["grad_acc_step"]
    grad_clip_thresh = train_config["optimizer"]["grad_clip_thresh"]
    total_step = train_config["step"]["total_step"]
    log_step = train_config["step"]["log_step"]
    save_step = train_config["step"]["save_step"]
    synth_step = train_config["step"]["synth_step"]
    val_step = train_config["step"]["val_step"]

    if resume_step >= total_step:
        print(
            "Training already finished: step {} / {}".format(resume_step, total_step)
        )
        return

    best_file = checkpoint_config["best_file"]
    last_file = get_last_file(train_config)
    best_path = os.path.join(ckpt_path, best_file)
    last_path = os.path.join(ckpt_path, last_file)
    best_score = float("inf")
    if os.path.exists(best_path):
        try:
            best_score = torch.load(
                best_path, map_location=device, weights_only=False
            ).get("val_loss", float("inf"))
        except Exception:
            pass

    # Discriminators live only in the `last` checkpoint (never in `best`, whose
    # score is pure L1 and must stay comparable to a non-GAN run). Restoring
    # them just avoids re-learning them from scratch on a resume.
    if resume_step > 0 and os.path.exists(last_path):
        try:
            _ck = torch.load(last_path, map_location=device, weights_only=False)
            if "adv" in _ck:
                Adv.load_state_dict(_ck["adv"])
                Adv.d_optimizer.load_state_dict(_ck["adv_optimizer"])
                print("Resumed discriminators from {} (step {})".format(last_file, _ck.get("step")))
        except Exception as e:
            print("Could not resume discriminators: {}".format(e))

    outer_bar = tqdm(total=total_step, desc="Training", position=0)
    outer_bar.n = resume_step
    outer_bar.update()

    while True:
        inner_bar = tqdm(total=len(loader), desc="Epoch {}".format(epoch), position=1)
        for batchs in loader:
            for batch in batchs:
                batch = to_device(batch, device)

                output = model(*(batch[2:]))

                losses = Loss(batch, output)
                total_loss = losses[0]

                adv_msg = ""
                if Adv.enabled and step >= Adv.start_step:
                    valid = ~output[5]
                    T = output[0].shape[1]
                    mel_real = batch[5][:, :T, :]
                    # Regularise the mel that actually reaches the vocoder
                    # (post-postnet), not the pre-postnet projection. The
                    # residual added by the postnet is exactly the channel an
                    # unconstrained generator uses to inject top-band junk,
                    # so judging the pre-postnet output leaves the loophole
                    # open -- and it also measures the wrong signal for the
                    # high-band brake below.
                    mel_fake = output[1][:, :T, :]
                    pitch = batch[8][:, :T]
                    labels = build_phone_frames(batch[2], batch[3], batch[10], T)
                    Adv.set_step(step)
                    d_loss, d_gap = Adv.d_step(
                        mel_real.detach(), mel_fake.detach(), pitch, labels, step
                    )
                    g = Adv.g_terms(mel_real, mel_fake, pitch, labels, valid)
                    adv_scale = Adv.scale(step)
                    total_loss = total_loss + adv_scale * g["total"]

                    w_adv_eff = Adv.w_adv * Adv.gain
                    hi_gap_ema = Adv._hi_ema or 0.0
                    if Adv.auto_gain and step % Adv.ctrl_every == 0:
                        # Noise-injection canary. When the generator starts
                        # chasing the discriminator instead of the data it
                        # typically buys acceptance with extra energy in the
                        # top mel bands, which HiFi-Gan renders as hiss.
                        #
                        # Masked by `valid`: mel_real is zero on padding while
                        # mel_fake carries the generator's own output there, so
                        # an unmasked mean over a ragged batch is dominated by
                        # padding and the brake fires on noise. `[:, :, -n:]`
                        # selects the top mel *bands*, so the frame mask has to
                        # broadcast over that dim -- (B, T, 1), not (B, n, 1).
                        # Only computed on control steps, which is the only place
                        # it is read; it costs a device sync.
                        w = valid.unsqueeze(-1).float()
                        diff = mel_fake[:, :, -Adv.hi_bands:] - \
                            mel_real[:, :, -Adv.hi_bands:]
                        hi_gap = ((diff * w).sum()
                                  / (w.sum() * Adv.hi_bands).clamp(min=1.0)).item()
                        Adv.update_gain(d_gap.item(), hi_gap)
                        hi_gap_ema = Adv._hi_ema or 0.0
                    adv_msg = (
                        ", D: {:.4f}, G Adv: {:.4f}, FM: {:.4f}, "
                        "Phone: {:.4f}, Pitch: {:.4f}, Scale: {:.2f}, "
                        "wAdv: {:.4f}, DGap: {:+.3f}, HiGap: {:+.3f}"
                    ).format(
                        d_loss.item(), g["adv"].item(), g["fm"].item(),
                        g["phone"].item(), g["pitch"].item(), adv_scale,
                        w_adv_eff, Adv._gap_ema or 0.0, hi_gap_ema
                    )

                total_loss = total_loss / grad_acc_step
                total_loss.backward()
                if step % grad_acc_step == 0:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip_thresh)

                    optimizer.step(step)
                    optimizer.zero_grad()

                if step % log_step == 0:
                    losses = [l.item() for l in losses]
                    message1 = "Step {}/{}, ".format(step, total_step)
                    message2 = "Total Loss: {:.4f}, Mel Loss: {:.4f}, Mel PostNet Loss: {:.4f}, Energy Loss: {:.4f}, LR: {:.2e}".format(
                        *(losses + [optimizer.current_lr])
                    ) + adv_msg

                    with open(os.path.join(train_log_path, "log.txt"), "a") as f:
                        f.write(message1 + message2 + "\n")

                    outer_bar.write(message1 + message2)

                    log(train_logger, step, losses=losses)

                if step % synth_step == 0:
                    fig, wav_reconstruction, wav_prediction, tag = synth_one_sample(
                        batch,
                        output,
                        vocoder,
                        model_config,
                        preprocess_config,
                    )
                    log(
                        train_logger,
                        fig=fig,
                        tag="Training/step_{}_{}".format(step, tag),
                    )
                    sampling_rate = preprocess_config["audio"]["sampling_rate"]
                    log(
                        train_logger,
                        audio=wav_reconstruction,
                        sampling_rate=sampling_rate,
                        tag="Training/step_{}_{}_reconstructed".format(step, tag),
                    )
                    log(
                        train_logger,
                        audio=wav_prediction,
                        sampling_rate=sampling_rate,
                        tag="Training/step_{}_{}_synthesized".format(step, tag),
                    )

                if step % val_step == 0:
                    model.eval()
                    message, loss_means = evaluate(
                        model, step, configs, val_logger, vocoder
                    )
                    with open(os.path.join(val_log_path, "log.txt"), "a") as f:
                        f.write(message + "\n")
                    outer_bar.write(message)

                    score = loss_means[1] + loss_means[2]
                    if score < best_score:
                        best_score = score
                        torch.save(
                            {
                                "model": model.module.state_dict(),
                                "optimizer": optimizer._optimizer.state_dict(),
                                "val_loss": score,
                                "step": step,
                            },
                            best_path,
                        )
                        best_message = (
                            "Step {}/{}, New best (mel+postnet={:.4f}) -> {}"
                        ).format(step, total_step, score, best_file)
                        outer_bar.write(best_message)
                        with open(
                            os.path.join(train_log_path, "log.txt"), "a"
                        ) as f:
                            f.write(best_message + "\n")

                    model.train()

                if step % save_step == 0:
                    torch.save(
                        {
                            "model": model.module.state_dict(),
                            "optimizer": optimizer._optimizer.state_dict(),
                            "adv": Adv.state_dict(),
                            "adv_optimizer": Adv.d_optimizer.state_dict(),
                            "step": step,
                        },
                        os.path.join(ckpt_path, last_file),
                    )

                if step == total_step:
                    quit()
                step += 1
                outer_bar.update(1)

            inner_bar.update(1)
        epoch += 1


if __name__ == "__main__":
    MODEL_CONFIG_PATH = "config/model.yaml"
    TRAIN_CONFIG_PATH = "config/train.yaml"

    parser = argparse.ArgumentParser(
        description="FastSinger preprocessing and training.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser(
        "train",
        help="extract mel, F0, energy and duration features from data, then train",
    )

    args = parser.parse_args()

    train_config = yaml.load(
        open(TRAIN_CONFIG_PATH, "r", encoding="utf-8"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(
        open(MODEL_CONFIG_PATH, "r", encoding="utf-8"), Loader=yaml.FullLoader
    )
    get_checkpoint_config(train_config, TRAIN_CONFIG_PATH)
    preprocess_config = merge_config(model_config, train_config)
    configs = (preprocess_config, model_config, train_config)

    preprocessor = Preprocessor(preprocess_config)
    preprocessor.build_from_path()
    del preprocessor
    run_train(args, configs)
