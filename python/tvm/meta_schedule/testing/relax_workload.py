import os

import torch
from diffusers import StableDiffusionPipeline
from tvm.relax.frontend.torch.dynamo import dynamo_capture_subgraphs


def get_stable_diffusion_model():
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")

    # save vae, unet, text_encoder, safety_checker
    dir = "/root/models"
    vae = torch.load(os.path.join(dir, "sd-v1-5-vae"))
    vae_params = list(vae.state_dict().values())
    mod = dynamo_capture_subgraphs(vae, *vae_params)


if __name__ == "__main__":
    get_stable_diffusion_model()
