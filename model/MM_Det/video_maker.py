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

import imageio

def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('-i', '--input-video', type=str, required=True, help='path to the input video file')
    # parser.add_argument('-o', '--output-video', type=str, required=True, help='path to the output video file')
    parser.add_argument('--ckpt', type=str, default='/home/dongryeol/MM-Det/weights/vqvae/model.pt', help='checkpoint path for vqvae')
    parser.add_argument('--device', type=str, default='cuda', help='device for reconstruction model')
    return parser.parse_args()


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

def process_frame(video_dir, )

def process_video(input_video, output_video, model, device, directory):
    # Open input video
    vc = cv2.VideoCapture(input_video)
    if not vc.isOpened():
        print("Error: Unable to open video file.")
        return
    
    # Get video properties
    fps = float(vc.get(cv2.CAP_PROP_FPS))
    frame_width = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Prepare the output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')


    writer = imageio.get_writer(output_video, fps=fps)
    
    recons_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    
    frame_count = 0
    while True:
        rval, frame = vc.read()
        if not rval:
            break
        
        frame_count += 1
        print(f"Processing frame {frame_count}")
        
        # Convert BGR frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert frame to PIL image
        img = Image.fromarray(frame_rgb)
        img = recons_transform(img).unsqueeze(0).to(device)
        
        # Get quantized reconstruction
        with torch.no_grad():
            recons, _, _ = model(img)
            if recons.shape != img.shape:
                recons = F.interpolate(recons, (img.shape[-2], img.shape[-1]), mode='nearest')
            # Denormalize and rearrange the tensor
            denorm_recons = denormalize_batch_t(recons, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            denorm_recons = rearrange(denorm_recons.squeeze(0).cpu().numpy(), 'c h w -> h w c')
            recons_img = np.clip(denorm_recons * 255.0, 0, 255).astype(np.uint8)


        # Ensure image is in BGR format before writing to output video
        if directory == 'fake':
            image_filename = f'/home/dongryeol/MM-Det/fake/frame_{frame_count}.jpg'
        elif directory == 'real':
            image_filename = f'/home/dongryeol/MM-Det/real/frame_{frame_count}.jpg'
        
        recons_img_pil = Image.fromarray(recons_img)
        recons_img_pil.save(image_filename)

        #recons_img_bgr = cv2.cvtColor(recons_img, cv2.COLOR_RGB2BGR)
        
        # Write reconstructed frame to output video
        writer.append_data(recons_img)


    writer.close()
    print(f"Reconstruction completed. Output saved at {output_video}")


if __name__ == '__main__':
    args = parse_args()
    
    # Load VQ-VAE model
    model = VectorQuantizedVAE(3, 256, 512)
    state_dict = torch.load(args.ckpt, map_location=args.device)
    model.load_state_dict(state_dict, strict=True)
    model.to(args.device)
    os.makedirs('/home/dongryeol/MM-Det/fake', exist_ok=True)
    os.makedirs('/home/dongryeol/MM-Det/real', exist_ok=True)
    
    # Process video
    #process_video('/home/dongryeol/MM-Det/화면 기록 2024-12-19 오후 9.06.18.mov', 'quantized_fake.mp4', model, 'cuda', 'fake')
    process_video('/home/dongryeol/MM-Det/opensora/0_real/화면 기록 2024-12-19 오후 9.07.29.mov', 'quantized_real.mp4', model, 'cuda', 'real')

