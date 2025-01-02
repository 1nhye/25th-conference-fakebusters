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
import imageio

from models import VectorQuantizedVAE


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data-root', type=str, default='data', help='data root for videos')
    parser.add_argument('-o', '--output', type=str, default='outputs/reconstruction_dataset/', help='output path')
    parser.add_argument('--ext', type=str, nargs='+', default=['mp4', 'avi', 'mpeg', 'wmv', 'mov', 'flv'], help='target extensions')
    parser.add_argument('--ckpt', type=str, default='weights/vqvae/model.pth', help='checkpoint path for vqvae')
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

def inference_frame_maker(mother_dir, video_dir):
    prefix = video_dir.split('.')[0]
    project_dir = os.path_join(mother_dir, prefix)
    original_dir = os.path_join(mother_dir, prefix, 'original')
    recon_dir = os.path_join(mother_dir, prefix, 'recon')

    os.makedir(project_dir)
    os.makedir(original_dir, exist_ok = True)
    os.makedir(recon_dir, exist_ok = True)
    model = VectorQuantizedVAE(3, 256, 512)
    state_dict = torch.load('weights/vqvae/model.pt', map_location=args.device)
    model.load_state_dict(state_dict, strict=True)
    model.to(args.device)
    vc = cv2.VideoCapture(video_dir)
    if not vc.isOpened():
        print("Error: Unable to open video file.")
        return
    
    # Get video properties
    fps = float(vc.get(cv2.CAP_PROP_FPS))
    frame_width = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Prepare the output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')



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

        origin_img_name = f'/{original_dir}_frame_{frame_count}'
        recon_img_name = f'/{recon_dir}_frame_{frame_count}'
        
        # Convert BGR frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert frame to PIL image
        img = Image.fromarray(frame_rgb)
        img.save(origin_img_name)
        
        device = 'cuda'
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



        recons_img_pil = Image.fromarray(recons_img)
        recons_img_pil.save(recon_img_name)

        #recons_img_bgr = cv2.cvtColor(recons_img, cv2.COLOR_RGB2BGR)
        
        # Write reconstructed frame to output video




    print(f"Frame Reconstruction completed. Output saved at {prefix}/{original_dir}")


   
if __name__ == '__main__':
    args = parse_args()
    cls_folders = list(filter(lambda p: os.path.isdir(os.path.join(args.data_root, p)), sorted(os.listdir(args.data_root))))
    recons_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    print(f'Find {len(cls_folders)} classes: {cls_folders}')
    model = VectorQuantizedVAE(3, 256, 512)
    state_dict = torch.load('weights/vqvae/model.pt', map_location=args.device)
    model.load_state_dict(state_dict, strict=True)
    model.to(args.device)
    for idx, cls_folder in enumerate(cls_folders, 1):
        os.makedirs(os.path.join(args.output, cls_folder, 'original'), exist_ok=True)
        video_list = list(filter(lambda fn: os.path.splitext(fn)[-1].lower()[1:] in args.ext, sorted(os.listdir(os.path.join(args.data_root, cls_folder)))))
        #breakpoint()
        for video in tqdm(video_list, desc='Extracting original frames'):
            #breakpoint()
            video_path = os.path.join(args.data_root, cls_folder, video)
            vc = cv2.VideoCapture(video_path)
            if vc.isOpened():
                rval, frame = vc.read()
            else:
                rval = False
            c = 1
            while rval:
                cv2.imwrite(os.path.join(args.output, cls_folder, 'original', f'{os.path.splitext(os.path.basename(video))[0]}_{c}.jpg'), frame)
                c = c + 1
                rval, frame = vc.read()
            vc.release()
        os.makedirs(os.path.join(args.output, cls_folder, 'recons'), exist_ok=True)
        for image_fn in tqdm(sorted(os.listdir(os.path.join(args.output, cls_folder, 'original'))), desc='Reconstruction'):
            img = Image.open(os.path.join(args.output, cls_folder, 'original', image_fn)).convert('RGB')
            img = recons_transform(img).unsqueeze(0)
            with torch.no_grad():
                img = img.to(args.device)
                recons, _, _ = model(img)
                if recons.shape != img.shape:
                    recons = F.interpolate(recons, (img.shape[-2], img.shape[-1]), mode='nearest')
                denorm_recons = denormalize_batch_t(recons, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                denorm_recons = rearrange(denorm_recons.squeeze(0).cpu().numpy(), 'c h w -> h w c')
                recons_img = np.uint8(denorm_recons * 255.)
                Image.fromarray(recons_img).save(os.path.join(args.output, cls_folder, 'recons', image_fn))
        print(f'Finished {idx}/{len(cls_folders)}')
    print(f'Finished. Results saved at {args.output}')