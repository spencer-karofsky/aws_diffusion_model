import torch
import torch.nn.functional as F

from dalle2.models.prior import Prior
from dalle2.models.decoder import Decoder
from dalle2.models.clip_encoding import CLIPEncoder
from dalle2.sampling.noise_scheduler import NoiseScheduler
from dalle2.sampling.ddim_sampling import DecoderDDIMSampler
from typing import Tuple

def device():
    if torch.cuda.is_available(): return torch.device('cuda')
    if torch.backends.mps.is_available(): return torch.device('mps')
    return torch.device('cpu')


@torch.no_grad()
def eps_with_cfg(prior, z_txt, t, z_t, scale, null_txt):
    eps_c = prior(z_txt=z_txt,    t=t, z_T=z_t)
    eps_u = prior(z_txt=null_txt, t=t, z_T=z_t)
    return eps_u + scale * (eps_c - eps_u)


@torch.no_grad()
def prior_short_window_ddim_cfg(
    prior, scheduler, z_txt, null_txt,
    *, start_t=20, steps=10, cfg_scale=3.0, device="cpu",
    z_img_true=None, init_from_text=True
):
    """
    Deterministic short-window DDIM with CFG in CLIP space.
    """
    D = z_txt.shape[-1]

    ts = torch.linspace(start_t, 0, steps, device=device).round().long().tolist()
    ts = sorted({int(v) for v in ts}, reverse=True)
    if ts[-1] != 0: ts.append(0)

    if init_from_text:
        if z_img_true is not None:
            denom = (z_txt.norm()**2 + 1e-8)
            W = (z_img_true.transpose(0,1) @ z_txt) / denom
            z_guess = (z_txt @ W.transpose(0,1))
        else:
            z_guess = z_txt
        z_guess = F.normalize(z_guess, dim=-1)

        t0 = torch.tensor([start_t], device=device)
        a_bar0 = scheduler.get_alpha_bar(t0).view(1,1)
        eps0 = torch.randn(1, D, device=device)
        z_t = torch.sqrt(a_bar0) * z_guess + torch.sqrt(1 - a_bar0) * eps0
    else:
        z_t = torch.randn(1, D, device=device)

    for i, tt in enumerate(ts):
        t = torch.tensor([tt], device=device)
        a_bar_t = scheduler.get_alpha_bar(t).view(1,1)
        sqrt_ab = torch.sqrt(a_bar_t)
        sqrt_1mab = torch.sqrt(1.0 - a_bar_t).clamp_min(1e-6)

        eps_hat = eps_with_cfg(prior, z_txt, t, z_t, cfg_scale, null_txt)

        # Per-step normalization
        eps_hat = eps_hat - eps_hat.mean(dim=-1, keepdim=True)
        eps_hat = eps_hat / eps_hat.std(dim=-1, keepdim=True).clamp_min(1e-6)

        x0_hat = (z_t - sqrt_1mab * eps_hat) / sqrt_ab
        x0_hat = F.normalize(x0_hat, dim=-1)
        eps_impl = (z_t - sqrt_ab * x0_hat) / sqrt_1mab

        if tt == 0:
            z_t = x0_hat
            break

        t_prev = torch.tensor([ts[i+1]], device=device)
        a_bar_prev = scheduler.get_alpha_bar(t_prev).view(1,1)
        sqrt_ab_prev = torch.sqrt(a_bar_prev)
        sqrt_1mab_prev = torch.sqrt(1.0 - a_bar_prev).clamp_min(1e-6)

        z_t = sqrt_ab_prev * x0_hat + sqrt_1mab_prev * eps_impl

    return F.normalize(z_t, dim=-1)


class DALLe2Text2Image:
    def __init__(
            self,
            prior_path: str,
            decoder_path: str,
            prior_T: int,
            start_T: int,
            steps_prior: int,
            cfg_scale: int,
            decoder_t: int,
            steps_decoder: int
    ):
        """
        Initializes 
        """
        self.dev = device()

        # Save params
        self.steps_prior = steps_prior
        self.cfg_scale = cfg_scale
        self.start_T = start_T

        # Schedulers
        self.prior_sched = NoiseScheduler(T=prior_T,   schedule_type='cosine')
        self.decoder_sched = NoiseScheduler(T=decoder_t, schedule_type='cosine')
        self.prior_sched.alpha_bar_t   = self.prior_sched.alpha_bar_t.to(self.dev)
        self.decoder_sched.alpha_bar_t = self.decoder_sched.alpha_bar_t.to(self.dev)

        # Models
        self.prior = Prior(device=self.dev, debug=False, T=prior_T, num_inference_steps=steps_prior).to(self.dev).eval()
        self.decoder = Decoder(device=self.dev, debug=False).to(self.dev).eval()

        self.prior.load_state_dict(torch.load(prior_path, map_location=self.dev), strict=True)

        self.decoder.load_state_dict(torch.load(decoder_path, map_location=self.dev), strict=False)

        # CLIP + null text (empty string)
        self.clip = CLIPEncoder().to(self.dev).eval()
        self.null_txt = self.clip.encode_text(['']).to(self.dev)

        # Decoder sampler
        self.decoder_sampler = DecoderDDIMSampler(self.decoder_sched, num_inference_steps=steps_decoder)
        self.decoder.sampler = self.decoder_sampler

    @torch.no_grad()
    def text_to_image(
            self,
            prompt: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate image conditioned by text prompt

        Args:
            prompt: the conditioning text prompt
        
        Returns:
            generated image, predicted image embedding
        """
        z_txt = self.clip.encode_text([prompt]).to(self.dev)

        z_img_hat = prior_short_window_ddim_cfg(
            self.prior, self.prior_sched, z_txt, self.null_txt,
            start_t=self.start_T, steps=self.steps_prior, cfg_scale=self.cfg_scale,
            device=self.dev, z_img_true=None, init_from_text=True
        )

        z_img_hat = z_img_hat / (z_img_hat.norm(dim=-1, keepdim=True) + 1e-8)

        img = self.decoder.sample(
            z_img=z_img_hat,
            sampler=self.decoder_sampler
        )

        return img, z_img_hat