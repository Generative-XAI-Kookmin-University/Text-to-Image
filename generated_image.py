import torch
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from PIL import Image
import os
from tqdm import tqdm

def load_model_from_config(config, ckpt):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    model = instantiate_from_config(config.model)
    model.load_state_dict(pl_sd["state_dict"], strict=False)
    model = model.cuda()
    model.eval()
    return model

def load_prompts_from_file(prompt_file, num_prompts):
    prompts = []
    with open(prompt_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(line)
            if len(prompts) >= num_prompts:
                break
    return prompts

def generate_images(
    config_path,
    model_path,
    prompt_file,
    num_images=10,
    ddim_steps=50,
    ddim_eta=1.0,
    save_dir='./generated_images'
):
    os.makedirs(save_dir, exist_ok=True)
    config = OmegaConf.load(config_path)
    model = load_model_from_config(config, model_path)
    sampler = DDIMSampler(model)

    prompts = load_prompts_from_file(prompt_file, num_images)

    for idx, prompt in tqdm(enumerate(prompts)):
        with torch.no_grad():
            c = model.cond_stage_model([prompt])
            shape = [
                model.model.diffusion_model.in_channels,
                model.model.diffusion_model.image_size,
                model.model.diffusion_model.image_size
            ]
            samples, _ = sampler.sample(
                S=ddim_steps,
                conditioning=c,
                batch_size=1,
                shape=shape,
                eta=ddim_eta,
                verbose=False
            )
            x_samples = model.decode_first_stage(samples)
            img = (x_samples[0] + 1) / 2
            arr = (img.permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')
            Image.fromarray(arr).save(os.path.join(save_dir, f'sample_{idx:05d}.png'))
            #print(f"Saved: {os.path.join(save_dir, f'sample_{idx:05d}.png')}")

if __name__ == "__main__":
    generate_images(
        config_path="/root/Text-to-Image/configs/latent-diffusion/txt2img/txt2img-sdv1.yaml",
        model_path="/root/Text-to-Image/pretrained/epoch=004-step=500000.ckpt",
        prompt_file="/root/coco/val_captions.txt",
        num_images=5000,  # 원하는 이미지 수
        ddim_steps=250,
        ddim_eta=1.0,
        save_dir="../coco/my_generated_images"
    )