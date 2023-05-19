import os

import numpy as np
import imageio
import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline

from dds import dds


def seed_everything(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main():
    seed_everything(0)
    pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-base").to('cuda')
    scheduler = DDIMScheduler.from_pretrained("stabilityai/stable-diffusion-2-1-base", subfolder="scheduler")
    os.makedirs(f"./results/", exist_ok=True)
    repeat_exp = 2

    prompt_lists = [
        ["A city view in the morning", "A city view at night with batman symbol in the sky"],
        ["A cat sitting on the floor", "A cat wearing a red tie"],
        ["A flamingo is roller-skating in the city", "A peacock is roller-skating in the city"],
    ]

    for prompt_id, (ref_prompt, update_prompt) in enumerate(prompt_lists):
        for exp_id in range(repeat_exp):
            print(
                f"Prompt {prompt_id}, Experiment {exp_id}, "
                f"ref_prompt: {ref_prompt}, update_prompt: {update_prompt}"
            )
            ref_img = pipe(ref_prompt, negative_prompt='low quality').images[0]
            ref_img.save(f'./results/{prompt_id:02d}_{exp_id:02d}_ref.png')
            ref_img = torch.as_tensor(np.array(ref_img)).float().cuda()/255
            upd_img = ref_img.clone()
            
            out_imgs = dds(ref_img, upd_img, scheduler, pipe, ref_prompt, update_prompt)

            imageio.mimwrite(f'./results/{prompt_id:02d}_{exp_id:02d}_out.mp4', out_imgs, fps=15)
            imageio.imwrite(f'./results/{prompt_id:02d}_{exp_id:02d}_upd.png', out_imgs[-1])
    return
  

if __name__ == '__main__':
    main()