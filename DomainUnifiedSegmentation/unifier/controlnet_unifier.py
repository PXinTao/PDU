"""ControlNet-based domain unifier wrapper.

This wrapper calls the Stage1&2 ControlNet diffusion code shipped in this repo.

It supports:
- single image reconstruction (target -> source-like style)
- user-provided hint (canny / HED / multi-channel)

Important:
- This requires GPU to be practical.
- You must provide Stable Diffusion weights (e.g., v1-5-pruned.ckpt) and
  a ControlNet backbone weight (e.g., control_sd15_canny.pth or your finetuned ckpt).
"""

from __future__ import annotations

import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import einops


@dataclass
class UnifierConfig:
    # Paths
    stage1_root: str  # path to "Stage1&2.Diffusion Model" folder
    cldm_yaml: str = "./models/cldm_v15.yaml"
    sd_ckpt: str = "./models/v1-5-pruned.ckpt"
    controlnet_ckpt: str = "./models/control_sd15_canny.pth"
    finetuned_ckpt: Optional[str] = None

    # Sampling
    ddim_steps: int = 50
    guidance_scale: float = 9.0
    strength: float = 1.0
    guess_mode: bool = False

    # Optional: img2img mode to better preserve original content.
    # When enabled, we encode the input image to latent and start denoising from
    # t_enc = img2img_strength * ddim_steps.
    use_img2img: bool = True
    img2img_strength: float = 0.6

    # Prompts
    prompt: str = "Ultrasound of breast"
    a_prompt: str = "realistic, best quality, extremely detailed"
    n_prompt: str = "fake 3D rendered image, bad anatomy, worst quality, low quality"

    # Resolution
    resolution: int = 512


class ControlNetUnifier:
    def __init__(self, cfg: UnifierConfig, device: str = "cuda"):
        self.cfg = cfg
        self.device = torch.device(device)

        # Add Stage1&2 folder to sys.path so we can import cldm/ldm/annotator
        stage_root = Path(cfg.stage1_root).resolve()
        if not stage_root.exists():
            raise FileNotFoundError(f"Stage1 root not found: {stage_root}")
        sys.path.insert(0, str(stage_root))

        # Deferred imports from ControlNet codebase
        from cldm.model import create_model, load_state_dict
        from cldm.ddim_hacked import DDIMSampler

        self._load_state_dict = load_state_dict
        self._DDIMSampler = DDIMSampler

        model = create_model(str(stage_root / cfg.cldm_yaml)).cpu()

        # Load SD base
        sd_states = load_state_dict(str(stage_root / cfg.sd_ckpt), location='cpu')
        model.load_state_dict(sd_states, strict=False)

        # Load ControlNet backbone
        cn_states = load_state_dict(str(stage_root / cfg.controlnet_ckpt), location='cpu')
        model.load_state_dict(cn_states, strict=False)

        # Optional: load finetuned ControlNet (your checkpoint after training)
        if cfg.finetuned_ckpt is not None and len(str(cfg.finetuned_ckpt)) > 0:
            ft_states = load_state_dict(str(cfg.finetuned_ckpt), location='cpu')
            model.load_state_dict(ft_states, strict=False)

        self.model = model.to(self.device)
        self.model.eval()

    def _seed(self, seed: int) -> int:
        if seed < 0:
            seed = random.randint(0, 65535)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        return seed

    def reconstruct(
        self,
        image_rgb: np.ndarray,
        hint_rgb: np.ndarray,
        prompt: Optional[str] = None,
        num_samples: int = 1,
        seed: int = -1,
    ) -> List[np.ndarray]:
        """Reconstruct an image given a hint.

        Args:
            image_rgb: HxWx3 uint8 (target-domain image)
            hint_rgb: HxWx3 float32 in [0,1] (control condition)

        Returns:
            list of reconstructed RGB uint8 images (len=num_samples)
        """
        cfg = self.cfg
        prompt = prompt or cfg.prompt

        # Resize to cfg.resolution
        H0, W0 = image_rgb.shape[:2]
        image = cv2.resize(image_rgb, (cfg.resolution, cfg.resolution), interpolation=cv2.INTER_AREA)
        hint = cv2.resize(hint_rgb, (cfg.resolution, cfg.resolution), interpolation=cv2.INTER_AREA)

        # Model expects hint in (B,3,H,W)
        hint = np.clip(hint, 0.0, 1.0).astype(np.float32)
        hint_chw = hint.transpose(2, 0, 1)  # 3,H,W
        hint_tensor = torch.stack([torch.from_numpy(hint_chw) for _ in range(num_samples)], dim=0).to(self.device)

        # Conditions
        sampler = self._DDIMSampler(self.model)
        # Ensure sampler schedules are created (needed for img2img path)
        sampler.make_schedule(ddim_num_steps=cfg.ddim_steps, ddim_eta=0.0, verbose=False)
        a_prompt = cfg.a_prompt
        n_prompt = cfg.n_prompt
        full_prompt = [prompt + ', ' + a_prompt] * num_samples

        self._seed(seed)

        cond = {
            "c_concat": [hint_tensor],
            "c_crossattn": [self.model.get_learned_conditioning(full_prompt)],
        }
        un_cond = {
            "c_concat": None if cfg.guess_mode else [hint_tensor],
            "c_crossattn": [self.model.get_learned_conditioning([n_prompt] * num_samples)],
        }

        # Control scales
        if cfg.guess_mode:
            self.model.control_scales = [cfg.strength * (0.825 ** float(12 - i)) for i in range(13)]
        else:
            self.model.control_scales = [cfg.strength] * 13

        # --- Sampling ---
        if cfg.use_img2img:
            # Encode input image to latent
            img_f = (image.astype(np.float32) / 127.5) - 1.0  # [-1,1]
            img_chw = img_f.transpose(2, 0, 1)
            x = torch.stack([torch.from_numpy(img_chw) for _ in range(num_samples)], dim=0).to(self.device)
            x = x.float()
            with torch.no_grad():
                posterior = self.model.encode_first_stage(x)
                z0 = self.model.get_first_stage_encoding(posterior)

            t_enc = int(round(cfg.img2img_strength * cfg.ddim_steps))
            t_enc = max(1, min(t_enc, cfg.ddim_steps - 1))
            t = torch.full((num_samples,), t_enc, device=self.device, dtype=torch.long)
            z_enc = sampler.stochastic_encode(z0, t)
            samples = sampler.decode(
                z_enc,
                cond,
                t_start=t_enc,
                unconditional_guidance_scale=cfg.guidance_scale,
                unconditional_conditioning=un_cond,
            )
        else:
            # Pure text+control generation from noise
            shape = (4, cfg.resolution // 8, cfg.resolution // 8)
            samples, _ = sampler.sample(
                cfg.ddim_steps,
                num_samples,
                shape,
                cond,
                verbose=False,
                unconditional_guidance_scale=cfg.guidance_scale,
                unconditional_conditioning=un_cond,
            )

        x_samples = self.model.decode_first_stage(samples)
        x_samples = (
            (einops.rearrange(x_samples, "b c h w -> b h w c") * 127.5 + 127.5)
            .detach()
            .cpu()
            .numpy()
            .clip(0, 255)
            .astype(np.uint8)
        )

        return [x_samples[i] for i in range(num_samples)]
