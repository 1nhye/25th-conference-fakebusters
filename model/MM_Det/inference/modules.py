import argparse
import cv2
import logging
import numpy as np
import os
import torch
from einops import rearrange
from tqdm import tqdm
from torch.nn import functional as F
from torchvision import transforms
from PIL import Image
import imageio
import sys
from tqdm import tqdm

# MM_Det 디렉토리를 sys.path에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from models import VectorQuantizedVAE
import torch
import os
import argparse
from tqdm import tqdm
from PIL import Image
import numpy as np
from transformers import AutoProcessor
from torchvision.transforms import Compose, CenterCrop, ToTensor, ToPILImage
from models import MMEncoder
from options.base_options import BaseOption
import logging

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, get_anyres_image_grid_shape
    


def get_dataset_meta(dataset_path):
    fns = sorted(os.listdir(dataset_path))
    meta = {}
    for fn in fns:
        data_id = fn.rsplit('_', maxsplit=1)[0]
        if data_id not in meta:
            meta[data_id] = 1
        else:
            meta[data_id] += 1
    return meta


def sample_by_interval(frame_count, interval=200):
    sampled_index = []
    count = 1
    while count <= frame_count:
        sampled_index.append(count)
        count += interval
    return sampled_index
    
 
 
def denormalize_batch_t(tensor, mean, std):
    """
    Denormalize a batch of tensors.
    """
    mean = torch.tensor(mean).view(1, -1, 1, 1).to(tensor.device)
    std = torch.tensor(std).view(1, -1, 1, 1).to(tensor.device)
    return tensor * std + mean

def inference_frame_maker(mother_dir, video_dir):
    prefix = os.path.splitext(os.path.basename(video_dir))[0]
    project_dir = os.path.join(mother_dir, prefix, '0_real')
    original_dir = os.path.join(project_dir, 'original')
    recon_dir = os.path.join(project_dir,  'recons')

    os.makedirs(project_dir, exist_ok=True)
    os.makedirs(original_dir, exist_ok=True)
    os.makedirs(recon_dir, exist_ok=True)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    logging.info(device)
    model = VectorQuantizedVAE(3, 256, 512)
    state_dict = torch.load('/root/frame_diffusion_detection/MM_Det/weights/vqvae/model.pt', map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)

    vc = cv2.VideoCapture(video_dir)
    if not vc.isOpened():
        print("Error: Unable to open video file.")
        return

    # Prepare transformations
    recons_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    batch_size = 32
    frame_count = 0
    frame_batch = []
    original_images = []

    while True:
        rval, frame = vc.read()
        if not rval:
            break

        frame_count += 1
        print(f"Processing frame {frame_count}")
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Save the original image
        original_img_path = os.path.join(original_dir, f'frame_{frame_count}.jpg')
        Image.fromarray(frame_rgb).save(original_img_path)
        original_images.append(original_img_path)
        
        # Prepare the frame for reconstruction
        frame_batch.append(recons_transform(Image.fromarray(frame_rgb)).unsqueeze(0))

        # Process the batch
        if len(frame_batch) == batch_size:
            frame_batch_tensor = torch.cat(frame_batch, dim=0).to(device)
            with torch.no_grad():
                recons_batch, _, _ = model(frame_batch_tensor)
                recons_batch = F.interpolate(recons_batch, (frame_batch_tensor.shape[-2], frame_batch_tensor.shape[-1]), mode='nearest')
                denorm_batch = denormalize_batch_t(recons_batch, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

            # Save reconstructed images
            for idx, recons_tensor in enumerate(denorm_batch):
                recons_img = np.clip(rearrange(recons_tensor.cpu().numpy(), 'c h w -> h w c') * 255.0, 0, 255).astype(np.uint8)
                recons_img_pil = Image.fromarray(recons_img)
                recons_img_pil.save(os.path.join(recon_dir, f'frame_{frame_count - batch_size + idx + 1}.jpg'))

            frame_batch.clear()

    # Process remaining frames in the batch
    if frame_batch:
        frame_batch_tensor = torch.cat(frame_batch, dim=0).to(device)
        with torch.no_grad():
            recons_batch, _, _ = model(frame_batch_tensor)
            recons_batch = F.interpolate(recons_batch, (frame_batch_tensor.shape[-2], frame_batch_tensor.shape[-1]), mode='nearest')
            denorm_batch = denormalize_batch_t(recons_batch, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        # Save reconstructed images
        for idx, recons_tensor in enumerate(denorm_batch):
            recons_img = np.clip(rearrange(recons_tensor.cpu().numpy(), 'c h w -> h w c') * 255.0, 0, 255).astype(np.uint8)
            recons_img_pil = Image.fromarray(recons_img)
            recons_img_pil.save(os.path.join(recon_dir, f'frame_{frame_count - len(frame_batch) + idx + 1}.jpg'))


# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def LMM_inference(image_dir, config):
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

    prefix = image_dir.split('/')[-1]
    
    project_dir = os.path.join(config['output_dir'], prefix, '0_real')
    os.makedirs(project_dir, exist_ok=True)
    image_dir = os.path.join(image_dir, '0_real')

    logging.info(f"Project directory: {project_dir}")
    logging.info(f"Image directory: {image_dir}")

    try:
        folder_path = os.listdir(image_dir)
    except FileNotFoundError as e:
        logging.error(f"Image directory not found: {image_dir}")
        raise e

    logging.info(f"Folders in image directory: {folder_path}")

    output_dir = config['output_dir']
    output_fn = config['output_fn']
    config['lmm_ckpt'] = 'sparklexfantasy/llava-1.5-7b-rfrd'
    config['load_4bit'] = False

    logging.info("LMM - Model Loading...")
    model = MMEncoder(config)
    visual_model = model.model.get_vision_tower().vision_tower

    logging.info("LMM - Model loaded successfully.")

    with torch.inference_mode():
        for folder in tqdm(folder_path, desc="Processing folders"):
            type_dir = os.path.join(project_dir, folder)
            os.makedirs(type_dir, exist_ok=True)

            folder = os.path.join(image_dir, folder)
            try:
                img_path = os.listdir(folder)
            except FileNotFoundError as e:
                logging.error(f"Folder not found: {folder}")
                raise e

            dataset_meta = get_dataset_meta(folder)

            logging.info(f"Processing folder: {folder}")
            logging.info(f"Number of images: {len(img_path)}")
            logging.info(f"Dataset meta: {dataset_meta}")

            mm_representations = {}

            for data_id, count in tqdm(dataset_meta.items(), desc="Processing dataset meta"):
                logging.info(f"Data ID: {data_id}, Count: {count}")

                sampled_index = sample_by_interval(count, config['interval'])
                logging.info(f"Sampled indices for {data_id}: {sampled_index}")

                for index in sampled_index:
                    img_file = os.path.join(folder, f"{data_id}_{index}.jpg")
                    try:
                        img = Image.open(img_file).convert('RGB')
                    except FileNotFoundError as e:
                        logging.error(f"Image file not found: {img_file}")
                        continue

                    logging.info(f"Processing image: {img_file}")

                    try:

                        visual_features, mm_features = model(img)
                    except Exception as e:
                        logging.error(f"Error during model inference on image {img_file}: {e}")
                        continue

                    mm_layer_features = {}
                    for idx, layer in enumerate(model.selected_layers):
                        mm_layer_features[str(layer)] = mm_features[idx].cpu()

                    mm_representations[f"{data_id}_{index}.jpg"] = {
                        "visual": visual_features.squeeze(0).cpu(),
                        "textual": mm_layer_features
                    }

            output_path = os.path.join(type_dir, output_fn)
            logging.info(f"Saving representations to: {output_path}")

            try:
                torch.save(mm_representations, output_path)
                logging.info(f"Successfully saved to {output_path}")
            except Exception as e:
                logging.error(f"Error saving to {output_path}: {e}")
                raise e
