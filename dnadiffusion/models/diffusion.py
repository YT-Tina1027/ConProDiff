from functools import partial

import torch
import torch.nn.functional as F
from torch import nn

from dnadiffusion.utils.utils import default, extract, linear_beta_schedule


class Diffusion(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        timesteps: int,
        beta_start: float,
        beta_end: float,
    ):
        super().__init__()
        self.model = model
        self.timesteps = timesteps

        betas = linear_beta_schedule(timesteps, beta_start, beta_end)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer(
            "posterior_variance",
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )

    @property
    def device(self):
        return self.betas.device

    # =========================================================
    # Sampling entry
    # =========================================================

    @torch.no_grad()
    def sample(self, classes, shape, cond_weight, gc_bin=None):
        """
        shape for 1D DNA diffusion:
            (batch_size, 4, seq_length)

        classes:
            strength condition, shape [B]
            0 = unconditional
            1/2/3 = low/mid/high

        gc_bin:
            GC condition, shape [B]
            0 = unconditional GC
            1~num_gc_bins-1 = real GC bins
        """
        return self.p_sample_loop(
            classes=classes,
            gc_bin=gc_bin,
            image_size=shape,
            cond_weight=cond_weight,
            get_cross_map=False,
        )

    @torch.no_grad()
    def sample_cross(self, classes, shape, cond_weight, gc_bin=None):
        return self.p_sample_loop(
            classes=classes,
            gc_bin=gc_bin,
            image_size=shape,
            cond_weight=cond_weight,
            get_cross_map=True,
        )

    @torch.no_grad()
    def p_sample_loop(
        self,
        classes,
        image_size,
        cond_weight,
        gc_bin=None,
        get_cross_map=False,
    ):
        b = image_size[0]
        device = self.device

        img = torch.randn(image_size, device=device)  # [B, 4, L]

        imgs = []
        cross_images_final = []

        if classes is not None:
            classes = classes.to(device).long()

            if gc_bin is not None:
                gc_bin = gc_bin.to(device).long()
                if gc_bin.shape[0] != classes.shape[0]:
                    raise ValueError(
                        f"gc_bin batch size {gc_bin.shape[0]} != classes batch size {classes.shape[0]}"
                    )

            sampling_fn = partial(
                self.p_sample_guided,
                classes=classes,
                gc_bin=gc_bin,
                cond_weight=cond_weight,
            )
        else:
            sampling_fn = partial(self.p_sample)

        for i in reversed(range(0, self.timesteps)):
            out = sampling_fn(
                x=img,
                t=torch.full((b,), i, device=device, dtype=torch.long),
                t_index=i,
            )

            if isinstance(out, tuple):
                img, cross_matrix = out
            else:
                img, cross_matrix = out, None

            imgs.append(img.detach().cpu().numpy())

            if get_cross_map and cross_matrix is not None:
                cross_images_final.append(cross_matrix.detach().cpu().numpy())

        if get_cross_map:
            return imgs, cross_images_final

        return imgs

    # =========================================================
    # Unconditional sampling
    # =========================================================

    @torch.no_grad()
    def p_sample(self, x, t, t_index):
        betas_t = extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x.shape)

        model_pred = self.model(x, time=t)

        if isinstance(model_pred, tuple):
            model_pred = model_pred[0]

        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * model_pred / sqrt_one_minus_alphas_cumprod_t
        )

        if t_index == 0:
            return model_mean

        posterior_variance_t = extract(self.posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise

    # =========================================================
    # Conditional sampling with CFG
    # =========================================================

    @torch.no_grad()
    def p_sample_guided(
        self,
        x,
        classes,
        t,
        t_index,
        cond_weight,
        gc_bin=None,
    ):
        """
        x: [B, 4, L]
        classes: [B]
        gc_bin: [B] or None

        CFG 现在同时对 strength condition 和 GC condition 做：
        - conditional: classes, gc_bin
        - unconditional: classes=0, gc_bin=0
        """
        batch_size = x.shape[0]
        device = self.device

        t_double = t.repeat(2).to(device)
        x_double = x.repeat(2, 1, 1).to(device)  # [2B, 4, L]

        betas_t = extract(self.betas, t_double, x_double.shape, device)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod,
            t_double,
            x_double.shape,
            device,
        )
        sqrt_recip_alphas_t = extract(
            self.sqrt_recip_alphas,
            t_double,
            x_double.shape,
            device,
        )

        classes = classes.long().to(device)
        classes_cond = classes
        classes_uncond = torch.zeros_like(classes, device=device)
        classes_double = torch.cat([classes_cond, classes_uncond], dim=0)

        if gc_bin is not None:
            gc_bin = gc_bin.long().to(device)
            gc_bin_cond = gc_bin
            gc_bin_uncond = torch.zeros_like(gc_bin, device=device)
            gc_bin_double = torch.cat([gc_bin_cond, gc_bin_uncond], dim=0)
        else:
            gc_bin_double = None

        self.model.output_attention = True
        preds, cross_map_full = self.model(
            x_double,
            time=t_double,
            classes=classes_double,
            gc_bin=gc_bin_double,
        )
        self.model.output_attention = False

        cross_map = None
        if cross_map_full is not None:
            cross_map = cross_map_full[:batch_size]

        eps_cond = preds[:batch_size]
        eps_uncond = preds[batch_size:]

        # classifier-free guidance
        x_t = eps_uncond + cond_weight * (eps_cond - eps_uncond)

        model_mean = sqrt_recip_alphas_t[:batch_size] * (
            x - betas_t[:batch_size] * x_t / sqrt_one_minus_alphas_cumprod_t[:batch_size]
        )

        if t_index == 0:
            return model_mean, cross_map

        posterior_variance_t = extract(self.posterior_variance, t, x.shape, device)
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise, cross_map

    # =========================================================
    # Forward diffusion
    # =========================================================

    def q_sample(self, x_start, t, noise=None):
        device = self.device
        noise = default(noise, torch.randn_like(x_start, device=device))

        sqrt_alphas_cumprod_t = extract(
            self.sqrt_alphas_cumprod,
            t,
            x_start.shape,
            device,
        )
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod,
            t,
            x_start.shape,
            device,
        )

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    # =========================================================
    # Training loss
    # =========================================================

    def p_losses(
        self,
        x_start,
        t,
        classes,
        gc_bin=None,
        noise=None,
        loss_type="huber",
        p_uncond=0.1,
    ):
        """
        CFG training:
        - 随机一部分样本把 classes 置 0
        - 同时把 gc_bin 置 0

        这样 unconditional 样本就是：
            strength=0, gc_bin=0

        不能只 drop classes 不 drop gc_bin，
        否则模型还能从 GC 条件里偷信息。
        """
        device = self.device
        noise = default(noise, torch.randn_like(x_start, device=device))

        x_noisy = self.q_sample(
            x_start=x_start,
            t=t,
            noise=noise,
        )

        classes = classes.to(device).long()

        if gc_bin is not None:
            gc_bin = gc_bin.to(device).long()
            if gc_bin.shape[0] != classes.shape[0]:
                raise ValueError(
                    f"gc_bin batch size {gc_bin.shape[0]} != classes batch size {classes.shape[0]}"
                )

        # classifier-free guidance training
        drop_mask = torch.rand(classes.shape[0], device=device) < p_uncond

        classes_input = classes.clone()
        classes_input[drop_mask] = 0

        if gc_bin is not None:
            gc_bin_input = gc_bin.clone()
            gc_bin_input[drop_mask] = 0
        else:
            gc_bin_input = None

        predicted_noise = self.model(
            x_noisy,
            time=t,
            classes=classes_input,
            gc_bin=gc_bin_input,
        )

        if isinstance(predicted_noise, tuple):
            predicted_noise = predicted_noise[0]

        if loss_type == "l1":
            loss = F.l1_loss(noise, predicted_noise)
        elif loss_type == "l2":
            loss = F.mse_loss(noise, predicted_noise)
        elif loss_type == "huber":
            loss = F.smooth_l1_loss(noise, predicted_noise)
        else:
            raise NotImplementedError(f"Unknown loss_type: {loss_type}")

        return loss

    def forward(self, x, classes, gc_bin=None):
        """
        x: [B, 4, L]
        classes: [B]
        gc_bin: [B] or None
        """
        device = self.device

        classes = classes.to(device).long()

        if gc_bin is not None:
            gc_bin = gc_bin.to(device).long()

        b = x.shape[0]
        t = torch.randint(
            0,
            self.timesteps,
            (b,),
            device=device,
        ).long()

        return self.p_losses(
            x_start=x,
            t=t,
            classes=classes,
            gc_bin=gc_bin,
        )