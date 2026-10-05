import argparse
from collections import OrderedDict
import json
import os

import numpy as np
from scipy.io import wavfile
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils import weight_norm, remove_weight_norm
import yaml

from Maker import VOICELESS_INITIALS

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def phone_to_sequence(text):
    tokens = text.replace("{", " ").replace("}", " ").split()
    return [symbol_to_id[t] for t in tokens if t in symbol_to_id]


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


def get_model(configs, device):
    (preprocess_config, model_config, infer_config) = configs

    model = FastSinger(preprocess_config, model_config).to(device)
    ckpt_path = os.path.join(
        infer_config["path"]["model_path"],
        infer_config["checkpoint"]["infer_file"],
    )
    if not os.path.exists(ckpt_path):
        raise SystemExit("Checkpoint not found: {}".format(ckpt_path))
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])

    model.eval()
    return model


def get_vocoder(device, infer_config):
    vocoder_config = infer_config.get("vocoder") or {}
    missing = [key for key in ("path", "file") if not vocoder_config.get(key)]
    if missing:
        raise SystemExit(
            "Missing required config in {}: {}".format(
                INFER_CONFIG_PATH,
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


def get_mask_from_lengths(lengths, max_len=None):
    batch_size = lengths.shape[0]
    if max_len is None:
        max_len = torch.max(lengths).item()

    ids = torch.arange(0, max_len, device=lengths.device).unsqueeze(0).expand(batch_size, -1)
    mask = ids >= lengths.unsqueeze(1).expand(-1, max_len)

    return mask


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


def parse_spec(json_text):
    try:
        spec = json.loads(json_text)
    except json.JSONDecodeError as e:
        raise SystemExit("Invalid JSON: {}".format(e))
    if not isinstance(spec, dict):
        raise SystemExit("JSON must be an object")
    for key in ("pinyin", "bpm", "bars", "midi"):
        if key not in spec:
            raise SystemExit("Missing field: {}".format(key))

    pinyin = str(spec["pinyin"]).strip().lower()
    if not pinyin:
        raise SystemExit("pinyin must not be empty")
    bpm = float(spec["bpm"])
    bars = float(spec["bars"])
    midi = float(spec["midi"])
    if bpm <= 0:
        raise SystemExit("bpm must be > 0")
    if bars <= 0:
        raise SystemExit("bars must be > 0")
    num = int(spec.get("num", 4))
    den = int(spec.get("den", 4))
    if num <= 0 or den <= 0:
        raise SystemExit("invalid time signature num/den")

    curve_raw = spec.get("curve", ["+0.0"])
    if not isinstance(curve_raw, list) or len(curve_raw) == 0:
        raise SystemExit("curve must be a non-empty list")
    try:
        curve = [float(v) for v in curve_raw]
    except (TypeError, ValueError):
        raise SystemExit("curve values must be numbers like \"+0.1\" or \"-0.2\"")

    return {
        "pinyin": pinyin,
        "bpm": bpm,
        "bars": bars,
        "midi": midi,
        "num": num,
        "den": den,
        "curve": curve,
    }


def compute_frames(bpm, bars, num, den, sampling_rate, hop_length):
    seconds = bars * num * (60.0 / bpm) * (4.0 / den)
    frames = int(round(seconds * sampling_rate / hop_length))
    return max(frames, 1), seconds


def allocate_durations(phones, total_frames, mean_duration):
    if len(phones) == 1:
        return [total_frames]
    first = int(round(mean_duration.get(phones[0], 2.0)))
    first = max(0, min(first, total_frames - 1))
    return [first, total_frames - first]


def build_f0(curve, total_frames, midi, phones, durations):
    if total_frames <= 1 or len(curve) == 1:
        offset = np.full(total_frames, curve[0], dtype=np.float64)
    else:
        xs = np.linspace(0, total_frames - 1, len(curve))
        offset = np.interp(np.arange(total_frames), xs, curve)
    semi = midi + offset
    f0 = 440.0 * 2.0 ** ((semi - 69.0) / 12.0)
    pos = 0
    for phone, d in zip(phones, durations):
        if phone in VOICELESS_INITIALS:
            f0[pos : pos + d] = 0.0
        pos += d
    return f0.astype(np.float32)


MODEL_CONFIG_PATH = "config/model.yaml"
INFER_CONFIG_PATH = "config/infer.yaml"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FastSinger inference.")
    parser.add_argument(
        "json_text",
        type=str,
        help='JSON like {"pinyin":"zhuang","bpm":120,"bars":1,"midi":60,"curve":["+0.0","+0.1"]}',
    )
    args = parser.parse_args()

    model_config = yaml.load(open(MODEL_CONFIG_PATH, "r"), Loader=yaml.FullLoader)
    infer_config = yaml.load(open(INFER_CONFIG_PATH, "r"), Loader=yaml.FullLoader)
    preprocess_config = model_config
    configs = (preprocess_config, model_config, infer_config)

    checkpoint_config = infer_config.get("checkpoint") or {}
    missing = [key for key in ("infer_file",) if not checkpoint_config.get(key)]
    if missing:
        raise SystemExit(
            "Missing required config in {}: {}".format(
                INFER_CONFIG_PATH,
                ", ".join("checkpoint." + key for key in missing),
            )
        )
    energy_control = infer_config.get("energy_control", 1.0)

    spec = parse_spec(args.json_text)

    preprocessed_path = preprocess_config["path"]["preprocessed_path"]
    with open(
        os.path.join(preprocessed_path, "pinyin2phones.json"), "r", encoding="utf-8"
    ) as f:
        pinyin2phones = json.load(f)
    with open(os.path.join(preprocessed_path, "stats.json"), "r") as f:
        stats = json.load(f)

    pinyin = spec["pinyin"]
    if pinyin not in pinyin2phones:
        raise SystemExit(
            "Unknown pinyin: {} ({} pinyins available, run Maker.py train first)".format(
                pinyin, len(pinyin2phones)
            )
        )
    phones_seq = pinyin2phones[pinyin]

    sampling_rate = preprocess_config["audio"]["sampling_rate"]
    hop_length = preprocess_config["audio"]["hop_length"]

    total_frames, seconds = compute_frames(
        spec["bpm"], spec["bars"], spec["num"], spec["den"], sampling_rate, hop_length
    )
    durations = allocate_durations(
        phones_seq, total_frames, stats["phone_mean_duration"]
    )
    f0 = build_f0(
        spec["curve"],
        total_frames,
        spec["midi"],
        phones_seq,
        durations,
    )

    sequence = phone_to_sequence("{" + " ".join(phones_seq) + "}")
    if len(sequence) != len(phones_seq):
        missing = [p for p in phones_seq if p not in symbol_to_id]
        raise SystemExit("Phones not in symbol table: {}".format(missing))

    model = get_model(configs, device)
    vocoder = get_vocoder(device, infer_config)
    print("Checkpoint: {}".format(checkpoint_config["infer_file"]))

    texts = torch.from_numpy(np.array([sequence])).long().to(device)
    src_lens = torch.from_numpy(np.array([len(sequence)])).to(device)
    f0s = torch.from_numpy(np.array([f0])).float().to(device)
    d_targets = torch.from_numpy(np.array([durations])).long().to(device)

    with torch.no_grad():
        output = model(
            texts,
            src_lens,
            len(sequence),
            f0s=f0s,
            d_targets=d_targets,
            e_control=energy_control,
        )

    mel = output[1].transpose(1, 2)
    lengths = torch.tensor([total_frames]) * hop_length
    wavs = vocoder_infer(mel, vocoder, preprocess_config, lengths=lengths)
    wav = wavs[0].astype(np.float64) / 32768.0
    peak = np.abs(wav).max()
    if peak > 0:
        wav *= 0.92 / peak
    wav_i16 = (wav * 32767.0).astype(np.int16)

    os.makedirs("out", exist_ok=True)
    out_path = os.path.join(
        "out", "{}_{}.wav".format(pinyin, int(round(spec["midi"])))
    )
    wavfile.write(out_path, sampling_rate, wav_i16)

    print("Pinyin: {}".format(pinyin))
    print("Phones: {} -> durations: {} frames".format(phones_seq, durations))
    print(
        "Time: {:.2f}s ({} frames, {}/{}, {} bpm)".format(
            seconds, total_frames, spec["num"], spec["den"], spec["bpm"]
        )
    )
    print("MIDI: {} with curve {}".format(spec["midi"], spec["curve"]))
    print("Saved: {}".format(out_path))
