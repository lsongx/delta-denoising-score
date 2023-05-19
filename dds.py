from typing import List, Tuple
from tqdm import tqdm
import numpy as np

import torch
import diffusers


@torch.no_grad()
def get_text_embeds(prompt, pipe, device='cuda'):
    # prompt: [str]
    inputs = pipe.tokenizer(prompt, padding='max_length', 
                            max_length=pipe.tokenizer.model_max_length, return_tensors='pt')
    embeddings = pipe.text_encoder(inputs.input_ids.to(device))[0]
    return embeddings


def get_img_latent(img, pipe):
    if img.shape[-1] == 3:
        img = img.permute(2, 0, 1)[None]
    img = 2 * img - 1
    posterior = pipe.vae.encode(img).latent_dist
    latents = posterior.sample() * pipe.vae.config.scaling_factor
    return latents


def decode_latent(latents, pipe):
    latents = 1 / pipe.vae.config.scaling_factor * latents
    imgs = pipe.vae.decode(latents).sample
    imgs = (imgs / 2 + 0.5).clamp(0, 1)
    return imgs


def get_noise_map(noise_pred, guidance_scale=7.5):
    noise_pred_uncond, noise_pred_pos = noise_pred.chunk(2)
    noise_map = noise_pred_uncond + guidance_scale * (noise_pred_pos - noise_pred_uncond)
    return noise_map


def pred_noise_map(latents, pipe, scheduler, noise, t, embeddings, guidance_scale=7.5):
    latents_noisy = scheduler.add_noise(latents, noise, t)
    # pred upd noise
    latent_input = torch.cat([latents_noisy] * 2)
    noise_pred = pipe.unet(latent_input, t, encoder_hidden_states=embeddings).sample
    # get noise map
    noise_map = get_noise_map(noise_pred, guidance_scale)
    return noise_map


def dds(ref_img: torch.FloatTensor, 
        upd_img: torch.FloatTensor, 
        scheduler: diffusers.schedulers.DDIMScheduler, 
        pipe: diffusers.pipelines.StableDiffusionPipeline, 
        ref_prompt: torch.FloatTensor, 
        update_prompt: torch.FloatTensor, 
        t_range: Tuple[float] = (0.02, 0.98), 
        guidance_scale: float = 2.5,
        iters: int = 200, 
        logging_freq: int = 10) -> List[np.ndarray]:
    """
    Implements the DDS algorithm from https://delta-denoising-score.github.io/
    
    Args:
        ref_img: reference image
        upd_img: image to be updated
        scheduler: DDIM scheduler
        pipe: diffusion pipeline
        ref_prompt: reference prompt
        update_prompt: prompt for updating
        t_range: range of timesteps to sample from
        guidance_scale: scale for guidance
        iters: number of iterations
        logging_freq: logging frequency
        
    Returns:
        List of images, with the last one being the final updated image
    """

    num_train_timesteps = scheduler.config.num_train_timesteps
    min_step = int(num_train_timesteps * t_range[0])
    max_step = int(num_train_timesteps * t_range[1])
    alphas = scheduler.alphas_cumprod.to('cuda') # for convenience

    ref_latents = get_img_latent(ref_img, pipe)
    upd_latents = get_img_latent(upd_img, pipe)
    neg_embeddings = get_text_embeds("", pipe)
    ref_embeddings = torch.cat([neg_embeddings, get_text_embeds(ref_prompt, pipe)], dim=0)
    upd_embeddings = torch.cat([neg_embeddings, get_text_embeds(update_prompt, pipe)], dim=0)

    upd_latents_param = torch.nn.Parameter(upd_latents.clone(), requires_grad=True)
    init_lr = 2.0
    init_momentum = 0.1
    optim = torch.optim.SGD([upd_latents_param], lr=init_lr, momentum=init_momentum)

    all_log_imgs = []
    for i in tqdm(range(iters)):
        t = torch.randint(min_step, max_step + 1, [1], dtype=torch.long, device='cuda')

        with torch.no_grad():
            # add noise
            noise = torch.randn_like(ref_latents)
            ref_noise_map = pred_noise_map(ref_latents, pipe, scheduler, 
                                           noise, t, ref_embeddings, guidance_scale)
            upd_noise_map = pred_noise_map(upd_latents_param, pipe, scheduler, 
                                           noise, t, upd_embeddings, guidance_scale)

        grad = (1 - alphas[t]) * (upd_noise_map - ref_noise_map)
        grad = torch.nan_to_num(grad)
        optim.zero_grad()
        upd_latents_param.grad = grad
        optim.step()

        if (i+1) % 20 == 0:
            # learning rate decay
            for param_group in optim.param_groups:
                param_group['lr'] *= 0.8
                param_group['momentum'] = init_momentum + (0.99-init_momentum) * ((i+1)/iters)

        if i % logging_freq == 0:
            with torch.no_grad():
                new_img = decode_latent(upd_latents_param, pipe)
                # plt.imsave(f'./z{i:03d}.png', new_img[0].cpu().numpy().transpose(1, 2, 0))
                all_log_imgs.append(new_img[0].cpu().numpy().transpose(1, 2, 0))

    with torch.no_grad():
        new_img = decode_latent(upd_latents_param, pipe)
        # plt.imsave(f'./z{i:03d}.png', new_img[0].cpu().numpy().transpose(1, 2, 0))
        all_log_imgs.append(new_img[0].cpu().numpy().transpose(1, 2, 0))
        for i in range(len(all_log_imgs)):
            all_log_imgs[i] = (all_log_imgs[i] * 255).astype(np.uint8)

    return all_log_imgs
