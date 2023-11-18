import logging
from collections import OrderedDict
from os import environ
from typing import Union

import debugpy
import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions.normal import Normal

import modules.scripts as scripts
from modules import rng, script_callbacks
from modules.processing import (
    StableDiffusionProcessing,
    StableDiffusionProcessingTxt2Img,
)
from modules.script_callbacks import CFGDenoiserParams

logger = logging.getLogger(__name__)
logger.setLevel(environ.get("SD_WEBUI_LOG_LEVEL", logging.INFO))

debugpy.listen(5678)
# debugpy.wait_for_client()


def latent_dist(latents):
    mu, logvar = torch.chunk(latents, 2, dim=-1)
    return mu, torch.sqrt(torch.exp(logvar))


def latent_log_pdf(p, q):
    from torch.distributions.normal import Normal

    q_mu = q[..., : q.shape[-1] // 2]
    return Normal(*latent_dist(p)).log_prob(q_mu)


def dpo_loss(win_latents, lose_latents, orig_latents, beta=0.1):
    "TODO: compute eqn. 7 over CLIP latents (https://arxiv.org/abs/2305.18290)"
    return torch.mean(
        beta * latent_log_pdf(lose_latents, orig_latents)
        - beta * latent_log_pdf(win_latents, orig_latents)
    )


def dpo_loss_clip(win_latents, lose_latents, orig_latents):
    return torch.mean(
        F.huber_loss(lose_latents, orig_latents, reduction="none")
        - F.huber_loss(win_latents, orig_latents, reduction="none")
    )


# def dpo_loss_clip_simple(win_latents, lose_latents):
#     return lose_latents - win_latents


class LatentAscent(scripts.Script):
    latents: list[torch.Tensor | None] = [None, None]
    optimizer: torch.optim.Optimizer | None = None
    chosen_at: int = -1
    iteration: int = 0
    win_cond: torch.Tensor | None = None
    loss_cond: torch.Tensor | None = None
    opt_state: dict | None = None
    learning_rate: float = 0.1
    momentum: float = 0.9
    traces: list[torch.Tensor] = [None, None]
    # next_latents: list[torch.Tensor | None] = [None, None]

    # Extension title in menu UI
    def title(self):
        return "Latent Ascent"

    # Decide to show menu in txt2img or img2img
    def show(self, is_img2img):
        return scripts.AlwaysVisible

    # Setup menu ui detail
    def ui(self, is_img2img):
        with gr.Accordion("Latent Ascent", open=False):
            active = gr.Checkbox(
                value=False, default=False, label="Active", elem_id="la_active"
            )
            with gr.Row():
                winner = gr.Dropdown(
                    ["Left", "Right"], label="Winner", elem_id="la_winner"
                )
                select_winner = gr.Button("Choose", elem_id="la_select_winner")

                def _choose_winner():
                    if self.chosen_at == self.iteration:
                        return
                    intwinner = int(winner.value == "Right")
                    win_cond = self.latents[intwinner]
                    loss_cond = self.latents[1 - intwinner]
                    loss = dpo_loss_clip(win_cond, loss_cond, self.latents[0])
                    params = [win_cond, loss_cond]
                    grads = torch.autograd.grad(loss, params)
                    traces = [
                        g.add_((self.momentum * t) if t is not None else 0)
                        for g, t in zip(grads, self.traces)
                    ]
                    self.traces = traces
                    self.latents = [
                        v + t * -self.learning_rate
                        for v, t in zip(self.latents, traces)
                    ]
                    self.chosen_at = self.iteration
                    self.win_cond = win_cond
                    self.loss_cond = loss_cond

                select_winner.click(_choose_winner)

        self.infotext_fields = [
            (active, "LA Active"),
            (winner, "Winner"),
        ]
        self.paste_field_names = [
            "la_active",
            "la_winner",
            "la_select_winner",
        ]
        return [active, winner, select_winner]

    def before_process(self, p: StableDiffusionProcessing, *args):
        # Set batch size and batch count to 1 and 2 respectively to produce and A and B sample
        p.batch_size = 1
        p.n_iter = 2
        self.prompt_latents = []

    def process(self, p: StableDiffusionProcessing, *args):
        for i in range(len(p.all_seeds)):
            p.all_seeds[i] = p.all_seeds[0]
            p.all_subseeds[i] = p.all_subseeds[0]

    def before_process_batch(self, p, *args, **kwargs):
        self.create_hook(p, kwargs["batch_number"])

    def create_hook(self, p, batch_num, *args, **kwargs):
        # Use lambda to call the callback function with the parameters to avoid global variables
        y = lambda params: self.on_cfg_denoiser_callback(
            params,
            batch_num=batch_num,
        )

        logger.debug("Hooked callbacks")
        script_callbacks.on_cfg_denoiser(y)
        script_callbacks.on_script_unloaded(self.unhook_callbacks)

    def postprocess_batch(self, p, *args, **kwargs):
        self.unhook_callbacks()

    def postprocess(self, p, processed, *args):
        self.iteration += 1

    def unhook_callbacks(self):
        logger.debug("Unhooked callbacks")
        script_callbacks.remove_current_script_callbacks()

    # def add_noise(self, y, gamma, noise_scale, psi, rescale=False):
    # """ CADS adding noise to the condition

    # Arguments:
    # y: Input conditioning
    # gamma: Noise level w.r.t t
    # noise_scale (float): Noise scale
    # psi (float): Rescaling factor
    # rescale (bool): Rescale the condition
    # """
    # y_mean, y_std = torch.mean(y), torch.std(y)
    # y = np.sqrt(gamma) * y + noise_scale * np.sqrt(1-gamma) * rng.randn_like(y)
    # if rescale:
    #     y_scaled = (y - torch.mean(y)) / torch.std(y) * y_std + y_mean
    #     if not torch.isnan(y_scaled).any():
    #         y = psi * y_scaled + (1 - psi) * y
    #     else:
    #         logger.debug("Warning: NaN encountered in rescaling")
    def add_noise(self, y, scale=0.1):
        rng.manual_seed(1337)
        return y + rng.randn_like(y) * scale

    def on_cfg_denoiser_callback(
        self,
        params: CFGDenoiserParams,
        batch_num,
    ):
        if self.iteration == 0:
            if params.sampling_step == 0:
                sampling_step = params.sampling_step
                text_cond: torch.Tensor = params.text_cond

                text_uncond: torch.Tensor = params.text_uncond
                if batch_num == 0:
                    new_cond = text_cond.requires_grad_()
                    params.text_cond = new_cond
                else:
                    new_cond = self.add_noise(text_cond, 0.025).requires_grad_()
                    params.text_cond = new_cond
                self.latents[batch_num] = params.text_cond
            else:
                params.text_cond = self.latents[batch_num]
        else:
            params.text_cond = self.latents[batch_num]
        # if self.optimizer is not None:
        #     self.optimizer.zero_grad()

    # def before_hr(self, p, *args):
    #     self.unhook_callbacks()
