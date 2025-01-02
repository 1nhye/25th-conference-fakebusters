import imageio
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from models import VectorQuantizedVAE
import os
import argparse
import cv2
import numpy as np
import os
import torch
from einops import rearrange
from tqdm import tqdm
from torch.nn import functional as F
from torchvision import transforms
from PIL import Image

from models import VectorQuantizedVAE


def denormalize_batch_t(img_t, mean, std):
    try:
        assert(len(mean) == len(std))
        assert(len(mean) == img_t.shape[1])
    except:
        print(f'Unmatched channels between image tensors and normalization mean and std. Got {img_t.shape[1]}, {len(mean)}, {len(std)}.')
    img_denorm = torch.empty_like(img_t)
    for t in range(img_t.shape[1]):
        img_denorm[:, t, :, :] = (img_t[:, t, :, :].clone() * std[t]) + mean[t]
    return img_denorm

def process_frame(frame, model, transform, device, frame_count):
    # Convert frame to PIL Image
    recons_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])


    # Transform the frame
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    img_tensor = recons_transform(img).unsqueeze(0).to(device)


    # Reconstruct the frame using the model
    with torch.no_grad():
        recons, _, _ = model(img_tensor)
        if recons.shape != img_tensor.shape:
            recons = F.interpolate(recons, (img_tensor.shape[-2], img_tensor.shape[-1]), mode='nearest')
        denorm_recons = denormalize_batch_t(recons, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        denorm_recons = rearrange(denorm_recons.squeeze(0).cpu().numpy(), 'c h w -> h w c')
        recons_img = np.clip(denorm_recons * 255.0, 0, 255).astype(np.uint8)

    # Save the frame for debugging
    save_path = f"debug_frames/frame_{frame_count:04d}.jpg"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)  # Ensure directory exists
    Image.fromarray(recons_img).save(save_path)
    print(f"Saved frame {frame_count} to {save_path}")

    return recons_img

def process_video(input_video, output_video, model, device):
    # Define the frame transformation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    # Read input video
    reader = imageio.get_reader(input_video, format="ffmpeg")
    fps = reader.get_meta_data()["fps"]

    writer = imageio.get_writer(output_video, fps=fps)

    frame_count = 0
    for frame in reader:
        #breakpoint()
        frame_count += 1
        print(f"Processing frame {frame_count}")

        # Process each frame
        processed_frame = process_frame(frame, model, transform, device, frame_count)
        writer.append_data(processed_frame)

    reader.close()
    writer.close()
    print(f"Video processing completed. Output saved to {output_video}")

# Example usage
if __name__ == "__main__":
    input_video_path = '/home/dongryeol/MM-Det/opensora/1_fake/-VeL80V8sa8.mp4'
    output_video_path = 'quantized_output.mp4'

    # Load VQ-VAE model
    model = VectorQuantizedVAE(3, 256, 512)
    state_dict = torch.load('/home/dongryeol/MM-Det/weights/vqvae/model.pt', map_location="cuda")
    model.load_state_dict(state_dict)
    model.to("cuda")
    model.eval()

    process_video(input_video_path, output_video_path, model, "cuda")
