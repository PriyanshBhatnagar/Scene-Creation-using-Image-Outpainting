import math
import os
import random
import PIL
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, PNDMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer
import fastcore.all as fc
from huggingface_hub import notebook_login
from pathlib import Path
import torchvision.transforms.functional as tf


def login_huggingface():
    """Login to huggingface account to load pretrained weights"""
    if not (Path.home() / '.huggingface' / 'token').exists():
        notebook_login()

class TextualInversionDataset(Dataset):
    """Dataset class for Textual Inversion"""
    def __init__(
        self,
        tokenizer,
        images,
        learnable_property="object",
        size=512,
        repeats=100,
        interpolation=Image.BICUBIC,
        flip_p=0.5,
        set="train",
        placeholder_token="*"
    ):
        fc.store_attr()
        self.num_images = len(images)
        if set == "train":
            self._length = self.num_images * repeats
        self.templates = style_templates if learnable_property == "style" else TEMPLATES
        self.flip_transform = transforms.RandomHorizontalFlip(p=self.flip_p)

    def __len__(self):
        return self.num_images

    def __getitem__(self, i):
        """Return one image along with the tokens"""
        image = tf.to_tensor(self.images[i % self.num_images]) * 2 - 1
        text = random.choice(self.templates).format(self.placeholder_token)
        ids = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt"
        )
        return {"input_ids": ids.input_ids[0], "pixel_values": image}


def get_dataset_attributes():
    """Get the attributes and data from our dataset"""

    path = Path('/data')
    paths = list(path.iterdir())
    images = [Image.open(p).resize((512, 512), resample=Image.BICUBIC).convert("RGB") for p in paths]

    what_to_teach = "object"
    placeholder_token = "Geisel Library"
    initializer_token = "Library"

    # Define the templates of the text prompts on which to train
    TEMPLATES = [
        "a photo of a {}",
        "a rendering of a {}",
        "a cropped photo of the {}",
        "the photo of a {}",
        "a photo of a clean {}",
        "a photo of a dirty {}",
        "a dark photo of the {}",
        "a photo of my {}",
        "a photo of the cool {}",
        "a close-up photo of a {}",
        "a bright photo of the {}",
        "a cropped photo of a {}",
        "a photo of the {}",
        "a good photo of the {}",
        "a photo of one {}",
        "a close-up photo of the {}",
        "a rendition of the {}",
        "a photo of the clean {}",
        "a rendition of a {}",
        "a photo of a nice {}",
        "a good photo of a {}",
        "a photo of the nice {}",
        "a photo of the small {}",
        "a photo of the weird {}",
        "a photo of the large {}",
        "a photo of a cool {}",
        "a photo of a small {}",
    ]

    return images, what_to_teach, placeholder_token, initializer_token, TEMPLATES


def training_function(
    text_encoder,
    vae,
    unet,
    train_batch_size,
    gradient_accumulation_steps,
    lr,
    max_train_steps,
    scale_lr,
    output_dir
):
    """Define the training function"""

    # The models are loaded with a precision of fp16 to reduce the memory footprint
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps, mixed_precision='fp16'
    )
    train_dataloader = create_dataloader(train_batch_size)
    if scale_lr:
        lr = (
            lr
            * gradient_accumulation_steps
            * train_batch_size
            * accelerator.num_processes
        )
    
    # Create an optimizer only on the text encoder parameters
    optimizer = torch.optim.AdamW(
        text_encoder.get_input_embeddings().parameters(), lr=lr
    )
    text_encoder, optimizer, train_dataloader = accelerator.prepare(
        text_encoder, optimizer, train_dataloader
    )

    # Move the corresponding parts to the GPU
    vae.to(accelerator.device).eval()
    unet.to(accelerator.device).eval()

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)
    total_batch_size = train_batch_size * accelerator.num_processes * gradient_accumulation_steps
    progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    global_step = 0

    loss_list = []

    # Run the training loop
    for epoch in range(num_train_epochs):
        # Set the text encoder to train mode
        text_encoder.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(text_encoder):

                # Get the latent space representation of the images in the batch
                latents = vae.encode(batch["pixel_values"]).latent_dist.sample().detach() * 0.18215
                
                # Generate noise
                noise = torch.randn(latents.shape).to(latents.device)
                bsz = latents.shape[0]

                # Define the timesteps for noise addition
                timesteps = torch.randint(
                    0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device
                ).long()

                # Add noise to the latent representation (Forward Diffusion)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]
                
                # Predict the amount of noise using noisy latent and encoder hidden state for conditioning
                # Denoising
                noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                
                # Compute the loss
                loss = F.mse_loss(noise_pred, noise, reduction="none").mean([1, 2, 3]).mean()
                accelerator.backward(loss)

                # We only want to optimize the concept embeddings
                grads = text_encoder.get_input_embeddings().weight.grad
                index_grads_to_zero = torch.arange(len(tokenizer)) != placeholder_token_id
                grads.data[index_grads_to_zero, :] = grads.data[index_grads_to_zero, :].fill_(0)
                
                # Optimize the weights
                optimizer.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

            # Save loss
            progress_bar.set_postfix(loss=loss.detach().item())
            loss_list.append(loss.detach().item())
            if global_step >= max_train_steps:
                break

    # Create a new Stable Diffusion Pipeline with safe weights
    pipeline = StableDiffusionPipeline(
        text_encoder=accelerator.unwrap_model(text_encoder),
        vae=vae,
        unet=unet,
        tokenizer=tokenizer,
        scheduler=PNDMScheduler(
            beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", skip_prk_steps=True
        ),
        safety_checker=StableDiffusionSafetyChecker.from_pretrained(
            "CompVis/stable-diffusion-safety-checker"
        ),
        feature_extractor=CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch32"),
    )

    # Save the learned weights and loss
    pipeline.save_pretrained(output_dir)
    learned_embeds = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[
        placeholder_token_id
    ]
    learned_embeds_dict = {placeholder_token: learned_embeds.detach().cpu()}
    torch.save(
        learned_embeds_dict, os.path.join(output_dir, "learned_embeds.bin")
    )

    loss_arr = np.array(loss_list)
    np.save('loss.npy', loss_arr)
    return pipeline, loss_arr


if __name__=="__main__":

    # Get the dataset attributes
    images, what_to_teach, placeholder_token, initializer_token, TEMPLATES = get_dataset_attributes()

    # Load the base pretrained model
    model_nm = "CompVis/stable-diffusion-v1-4"

    # Create the tokenizer and the new token to it
    tokenizer = CLIPTokenizer.from_pretrained(model_nm, subfolder="tokenizer")
    num_added_tokens = tokenizer.add_tokens(placeholder_token)
    token_ids = tokenizer.encode(initializer_token, add_special_tokens=False)
    initializer_token_id = token_ids[0]
    placeholder_token_id = tokenizer.convert_tokens_to_ids(placeholder_token)

    # Load all the individual components of Stable Diffusion
    text_encoder = CLIPTextModel.from_pretrained(model_nm, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(model_nm, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(model_nm, subfolder="unet")
    text_encoder.resize_token_embeddings(len(tokenizer))
    token_embeds = text_encoder.get_input_embeddings().weight.data
    token_embeds[placeholder_token_id] = token_embeds[initializer_token_id]

    # Freeze all parameters except for the token embeddings in text encoder
    tm = text_encoder.text_model
    for o in (vae, unet, tm.encoder, tm.final_layer_norm, tm.embeddings.position_embedding):
        for p in o.parameters():
            p.requires_grad = False

    # Create the training dataset
    train_dataset = TextualInversionDataset(
        images=images,
        tokenizer=tokenizer,
        size=512,
        placeholder_token=placeholder_token,
        repeats=100,
        learnable_property=what_to_teach,
        set="train"
    )

    # Create dataloader
    def create_dataloader(bs=1):
        return DataLoader(train_dataset, batch_size=bs, shuffle=True)

    # Create the noise scheduler describing how much noise to be added at each time step
    noise_scheduler = DDPMScheduler(
        beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000
    )

    # Call the training function to train the textual inversion model
    torch.manual_seed(42)
    pipeline, loss = training_function(
        text_encoder,
        vae,
        unet,
        train_batch_size=1,
        gradient_accumulation_steps=4,
        lr=5e-04,
        max_train_steps=2500,
        scale_lr=True,
        output_dir="learned-geisel",
    )

    # Compute the loss per epoch by averaging for batch
    loss_ep = []
    i=0
    while i < loss.shape[0]:
        loss_ep.append(np.mean(loss[i:i+4]))
        i += 4

    # Plot the loss function
    plt.plot(loss_ep, color='grey')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss")
