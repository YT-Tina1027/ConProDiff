import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F


def default(val, d):
    return val if val is not None else d


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class PreNorm1d(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.GroupNorm(1, dim)
        self.fn = fn

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


class LearnedSinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x):
        # x: [B]
        x = x.float().unsqueeze(1)  # [B, 1]
        freqs = x * self.weights.unsqueeze(0) * 2 * math.pi
        fouriered = torch.cat((x, freqs.sin(), freqs.cos()), dim=-1)
        return fouriered


class Block1d(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = nn.Conv1d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ResnetBlock1d(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = (
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_emb_dim, dim_out * 2)
            )
            if time_emb_dim is not None
            else None
        )

        self.block1 = Block1d(dim, dim_out, groups=groups)
        self.block2 = Block1d(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None

        if self.mlp is not None and time_emb is not None:
            time_emb = self.mlp(time_emb)          # [B, 2 * dim_out]
            time_emb = time_emb.unsqueeze(-1)      # [B, 2 * dim_out, 1]
            scale_shift = time_emb.chunk(2, dim=1) # (scale, shift)

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)

        return h + self.res_conv(x)


class LinearAttention1d(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()

        self.heads = heads
        self.dim_head = dim_head
        hidden_dim = heads * dim_head

        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Sequential(
            nn.Conv1d(hidden_dim, dim, 1),
            nn.GroupNorm(1, dim)
        )

    def forward(self, x):
        b, c, n = x.shape

        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = qkv

        h = self.heads
        d = self.dim_head

        q = q.view(b, h, d, n)
        k = k.view(b, h, d, n)
        v = v.view(b, h, d, n)

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * (d ** -0.5)

        context = torch.einsum("bhdn,bhen->bhde", k, v)
        out = torch.einsum("bhde,bhdn->bhen", context, q)
        out = out.reshape(b, h * d, n)

        return self.to_out(out)


class Downsample1d(nn.Module):
    def __init__(self, dim, dim_out=None):
        super().__init__()

        dim_out = default(dim_out, dim)
        self.conv = nn.Conv1d(
            dim,
            dim_out,
            kernel_size=4,
            stride=2,
            padding=1
        )

    def forward(self, x):
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim, dim_out=None):
        super().__init__()

        dim_out = default(dim_out, dim)
        self.conv = nn.ConvTranspose1d(
            dim,
            dim_out,
            kernel_size=4,
            stride=2,
            padding=1
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(
        self,
        dim: int,
        init_dim: int | None = None,
        dim_mults: list = [1, 2, 4],
        channels: int = 4,
        resnet_block_groups: int = 8,
        learned_sinusoidal_dim: int = 18,
        num_classes: int | None = 10,
        num_gc_bins: int | None = None,
        use_gc_condition: bool = True,
        output_attention: bool = False,
    ) -> None:
        """
        Conditional 1D UNet for DNA diffusion.

        条件设计：
        - classes:
            strength condition
            0 = unconditional
            1/2/3 = low/mid/high
        - gc_bin:
            GC condition
            0 = unconditional GC
            1~num_gc_bins-1 = real GC bins

        最终条件注入方式：
            t_emb = time_emb + label_emb(classes) + gc_emb(gc_bin)
        """
        super().__init__()

        self.channels = channels
        self.output_attention = output_attention

        self.num_classes = num_classes
        self.num_gc_bins = num_gc_bins
        self.use_gc_condition = use_gc_condition

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv1d(channels, init_dim, kernel_size=7, padding=3)

        dims = [init_dim, *(dim * m for m in dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock1d, groups=resnet_block_groups)

        time_dim = dim * 4
        sinu_pos_emb = LearnedSinusoidalPosEmb(learned_sinusoidal_dim)
        fourier_dim = learned_sinusoidal_dim + 1

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # strength condition embedding
        if num_classes is not None and num_classes > 0:
            self.label_emb = nn.Embedding(num_classes, time_dim)
        else:
            self.label_emb = None

        # GC condition embedding
        if use_gc_condition and num_gc_bins is not None and num_gc_bins > 0:
            self.gc_emb = nn.Embedding(num_gc_bins, time_dim)
        else:
            self.gc_emb = None

        self.downs = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        Residual(PreNorm1d(dim_in, LinearAttention1d(dim_in))),
                        Downsample1d(dim_in, dim_out)
                        if not is_last
                        else nn.Conv1d(dim_in, dim_out, 3, padding=1),
                    ]
                )
            )

        mid_dim = dims[-1]

        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm1d(mid_dim, LinearAttention1d(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        self.ups = nn.ModuleList([])

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                        block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                        Residual(PreNorm1d(dim_out, LinearAttention1d(dim_out))),
                        Upsample1d(dim_out, dim_in)
                        if not is_last
                        else nn.Conv1d(dim_out, dim_in, 3, padding=1),
                    ]
                )
            )

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv1d(dim, channels, kernel_size=1)

    def _match_length(self, x: torch.Tensor, skip: torch.Tensor):
        target_len = min(x.shape[-1], skip.shape[-1])

        if x.shape[-1] != target_len:
            x = x[..., :target_len]

        if skip.shape[-1] != target_len:
            skip = skip[..., :target_len]

        return x, skip

    def _build_condition_embedding(
        self,
        time: torch.Tensor,
        classes: torch.Tensor | None = None,
        gc_bin: torch.Tensor | None = None,
    ):
        """
        构建联合条件 embedding：
            time + strength label + GC bin

        注意：
        - classes 和 gc_bin 都应该是 LongTensor，shape=[B]
        - classes=0 是 unconditional strength
        - gc_bin=0 是 unconditional GC
        """
        t_emb = self.time_mlp(time)

        if classes is not None and self.label_emb is not None:
            classes = classes.long()
            t_emb = t_emb + self.label_emb(classes)

        if gc_bin is not None and self.gc_emb is not None:
            gc_bin = gc_bin.long()
            t_emb = t_emb + self.gc_emb(gc_bin)

        return t_emb

    def forward(
        self,
        x: torch.Tensor,
        time: torch.Tensor,
        classes: torch.Tensor | None = None,
        gc_bin: torch.Tensor | None = None,
    ):
        """
        x: [B, 4, L]
        time: [B]
        classes: [B] or None
        gc_bin: [B] or None

        返回：
        - diffusion predicted noise / x_start，shape=[B, 4, L]
        """
        orig_len = x.shape[-1]

        x = self.init_conv(x)
        residual = x.clone()

        t_emb = self._build_condition_embedding(
            time=time,
            classes=classes,
            gc_bin=gc_bin,
        )

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t_emb)
            h.append(x)

            x = block2(x, t_emb)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t_emb)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t_emb)

        for block1, block2, attn, upsample in self.ups:
            skip = h.pop()
            x, skip = self._match_length(x, skip)
            x = torch.cat((x, skip), dim=1)
            x = block1(x, t_emb)

            skip = h.pop()
            x, skip = self._match_length(x, skip)
            x = torch.cat((x, skip), dim=1)
            x = block2(x, t_emb)

            x = attn(x)
            x = upsample(x)

        x, residual = self._match_length(x, residual)
        x = torch.cat((x, residual), dim=1)

        x = self.final_res_block(x, t_emb)
        x = self.final_conv(x)

        cur_len = x.shape[-1]

        if cur_len > orig_len:
            x = x[..., :orig_len]
            cur_len = x.shape[-1]

        pad_len = orig_len - cur_len

        if pad_len > 0:
            x = F.pad(x, (0, pad_len))

        if self.output_attention:
            return x, None

        return x