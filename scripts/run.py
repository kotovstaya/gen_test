import argparse
import torch
import numpy as np
from einops import rearrange
from torch import autocast
from contextlib import nullcontext
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.extras import load_model_from_config
import clip
from PIL import Image
from huggingface_hub import hf_hub_download


ckpt = hf_hub_download(repo_id="lambdalabs/image-mixer", filename="image-mixer-pruned.ckpt")
config = hf_hub_download(repo_id="lambdalabs/image-mixer", filename="image-mixer-config.yaml")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = load_model_from_config(config, ckpt, device=device, verbose=False)
if torch.cuda.is_available():
    model = model.to(device).half()
else:
    model = model.to(device)


clip_model, preprocess = clip.load("ViT-L/14", device=device)
torch.cuda.empty_cache()


@torch.no_grad()
def get_im_c(im_path, clip_model):
    prompts = preprocess(im_path).to(device).unsqueeze(0)
    return clip_model.encode_image(prompts).float()


def to_im_list(x_samples_ddim):
    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
    ims = []
    for x_sample in x_samples_ddim:
        x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
        ims.append(Image.fromarray(x_sample.astype(np.uint8)))
    return ims


@torch.no_grad()
def sample(sampler, model, c, uc, scale, start_code, h=512, w=512, precision="autocast",ddim_steps=50, x0=None):
    ddim_eta=0.0
    precision_scope = autocast if precision=="autocast" else nullcontext
    with precision_scope("cuda"):
        shape = [4, h // 8, w // 8]
        samples_ddim, _ = sampler.sample(S=ddim_steps,
                                         x0=x0,
                                            conditioning=c,
                                            batch_size=c.shape[0],
                                            shape=shape,
                                            verbose=False,
                                            unconditional_guidance_scale=scale,
                                            unconditional_conditioning=uc,
                                            eta=ddim_eta,
                                            x_T=start_code)

        x_samples_ddim = model.decode_first_stage(samples_ddim)
    return to_im_list(x_samples_ddim)


def run(inps):
    scale, n_samples, seed, steps = 6, 1, 0, 10
    h = w = 640

    sampler = DDIMSampler(model)

    torch.manual_seed(seed)
    start_code = torch.randn(n_samples, 4, h//8, w//8, device=device)
    conds = []

    imgs = []
    for im in zip(*inps):
        this_cond = get_im_c(im, clip_model)
        imgs.append(im)
        conds.append(this_cond)
    conds = conds[1:]
    conds = torch.cat(conds, dim=0).unsqueeze(0)
    conds = conds.tile(n_samples, 1, 1)

    ims = sample(sampler, model, conds, 0*conds, scale, start_code, ddim_steps=steps, x0=imgs[0])
    return ims


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('image1')
    parser.add_argument('image2')

    args = parser.parse_args()

    ims = run(
        (
            Image.open(args.image1),
            Image.open(args.image2),
        )
    )
    for ix, img in enumerate(ims):
        img.save(f"/Users/anton.savinkov/Desktop/{ix}_img.png")

    # python run.py /Users/anton.savinkov/Desktop/boro.png /Users/anton.savinkov/Desktop/mike.png
