import time
from typing import Tuple, Optional

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision

from dalle2.models.prior import Prior
from dalle2.models.decoder import Decoder
from dalle2.models.clip_encoding import CLIPEncoder
from dalle2.sampling.noise_scheduler import NoiseScheduler
from dalle2.sampling.ddim_sampling import DecoderDDIMSampler

# -------------------------------------------------------------------------------------
# helpers
# -------------------------------------------------------------------------------------

def _dev() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# -------------------------------------------------------------------------------------
# PRIOR  •  short‑window DDIM + classifier‑free guidance (no per‑step ε normalisation!)
# -------------------------------------------------------------------------------------
@torch.no_grad()
def eps_with_cfg(prior: Prior, z_txt: torch.Tensor, t: torch.Tensor, z_t: torch.Tensor,
                 scale: float, null_txt: torch.Tensor) -> torch.Tensor:
    """Blend conditional / unconditional noise predictions."""
    eps_c = prior(z_txt=z_txt, t=t, z_T=z_t)
    eps_u = prior(z_txt=null_txt, t=t, z_T=z_t)
    return eps_u + scale * (eps_c - eps_u)


@torch.no_grad()
def prior_short_window_ddim_cfg(
    prior: Prior,
    scheduler: NoiseScheduler,
    z_txt: torch.Tensor,
    null_txt: torch.Tensor,
    *,
    start_t: int = 180,
    steps: int = 100,
    cfg_scale: float = 4.25,
    device: torch.device,
    init_from_text: bool = True,
) -> torch.Tensor:
    """Deterministic DDIM sampling in CLIP‑embedding space using CFG."""
    D = z_txt.shape[-1]

    # --- discrete timestep schedule --------------------------------------------------
    ts = torch.linspace(start_t, 0, steps, device=device).round().long().tolist()
    ts = sorted({int(v) for v in ts}, reverse=True)
    if ts[-1] != 0:
        ts.append(0)

    # --- initialise latent -----------------------------------------------------------
    if init_from_text:
        z_guess = F.normalize(z_txt, dim=-1)
        t0 = torch.tensor([start_t], device=device)
        a_bar0 = scheduler.get_alpha_bar(t0).view(1, 1)
        eps0 = torch.randn(1, D, device=device)
        z_t = torch.sqrt(a_bar0) * z_guess + torch.sqrt(1 - a_bar0) * eps0
    else:
        z_t = torch.randn(1, D, device=device)

    # --- reverse process -------------------------------------------------------------
    for i, tt in enumerate(ts):
        t = torch.tensor([tt], device=device)
        a_bar_t = scheduler.get_alpha_bar(t).view(1, 1)
        sqrt_ab      = torch.sqrt(a_bar_t)
        sqrt_1mab    = torch.sqrt(1.0 - a_bar_t).clamp_min(1e-6)

        eps_hat = eps_with_cfg(prior, z_txt, t, z_t, cfg_scale, null_txt)
        #eps_hat = prior(z_txt=z_txt, t=t, z_T=z_t)  # no CFG


        x0_hat = (z_t - sqrt_1mab * eps_hat) / sqrt_ab
        x0_hat = F.normalize(x0_hat, dim=-1)

        if tt == 0:  # finished
            return x0_hat

        # deterministic DDIM step
        t_next = torch.tensor([ts[i + 1]], device=device)
        a_bar_next = scheduler.get_alpha_bar(t_next).view(1, 1)
        sqrt_ab_next   = torch.sqrt(a_bar_next)
        sqrt_1mab_next = torch.sqrt(1.0 - a_bar_next).clamp_min(1e-6)
        eps_impl = (z_t - sqrt_ab * x0_hat) / sqrt_1mab
        z_t = sqrt_ab_next * x0_hat + sqrt_1mab_next * eps_impl

        

    # should never reach here
    return F.normalize(z_t, dim=-1)

# -------------------------------------------------------------------------------------
#  DALLE‑2  Text‑to‑Image  (Prior + Decoder)
# -------------------------------------------------------------------------------------
class DALLe2Text2Image:
    def __init__(
        self,
        *,
        prior_path: str,
        decoder_path: str,
        prior_T: int = 200,
        start_T: int = 180,
        steps_prior: int = 100,
        prior_cfg_scale: float = 7.0,
        decoder_T: int = 200,
        steps_decoder: int = 50,
        decoder_cfg_scale: float = 5.75,
    ) -> None:
        self.dev = _dev()

        # ─── schedulers ────────────────────────────────────────────────────────────
        self.prior_sched = NoiseScheduler(T=prior_T,   schedule_type="cosine")
        self.decoder_sched = NoiseScheduler(T=decoder_T, schedule_type="cosine")
        self.prior_sched.alpha_bar_t   = self.prior_sched.alpha_bar_t.to(self.dev)
        self.decoder_sched.alpha_bar_t = self.decoder_sched.alpha_bar_t.to(self.dev)

        # ─── models ────────────────────────────────────────────────────────────────
        self.prior = Prior(device=self.dev, T=prior_T ).eval().to(self.dev)
        self.decoder = Decoder(device=self.dev).eval().to(self.dev)
        self.prior.load_state_dict(torch.load(prior_path,   map_location=self.dev), strict=True)
        self.decoder.load_state_dict(torch.load(decoder_path, map_location=self.dev), strict=False)

        # ─── CLIP & null text embedding ────────────────────────────────────────────
        self.clip = CLIPEncoder().eval().to(self.dev)
        self.null_txt = F.normalize(self.clip.encode_text([""]).to(self.dev), dim=-1)

        # ─── decoder sampler (DDIM + CFG) ──────────────────────────────────────────
        self.decoder_sampler = DecoderDDIMSampler(
            self.decoder_sched,
            num_inference_steps = steps_decoder,
            guidance_scale = decoder_cfg_scale,
        )

        # ─── hyper‑params to reuse ─────────────────────────────────────────────────
        self.start_T = start_T
        self.steps_prior = steps_prior
        self.prior_cfg = prior_cfg_scale

    # -------------------------------------------------------------------------
    @torch.no_grad()
    def text_to_image(self, prompt: str, *, log_cosine: bool = True) -> torch.Tensor:
        z_txt = F.normalize(self.clip.encode_text([prompt]), dim=-1).to(self.dev)

        assert torch.allclose(z_txt.norm(dim=-1), torch.ones(1, device=z_txt.device)), "z_txt not normalized"
        assert torch.allclose(self.null_txt.norm(dim=-1), torch.ones(1, device=self.null_txt.device)), "null_txt not normalized"

        # PRIOR ➜ z_img_hat
        z_img_hat = prior_short_window_ddim_cfg(
            self.prior, self.prior_sched,
            z_txt, self.null_txt,
            start_t = self.start_T,
            steps = self.steps_prior,
            cfg_scale = self.prior_cfg,
            device = self.dev,
        )

        if log_cosine:
            cos = F.cosine_similarity(z_txt, z_img_hat, dim=-1).item()
            print(f"cos(text , img_hat) = {cos:+.4f}")
    
        # DECODER ➜ image
        img = self.decoder.sample(
            z_img   = z_img_hat,
            sampler = self.decoder_sampler,
        )
        #img, z_img_hat = pipe.text_to_image(prompt)
        vis = (img.clamp(-1,1)+1)/2
        return img, z_img_hat