import yaml
import logging
import argparse
import sys
import os
import torch

# Add the parent directory to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from classifier import Classifier
from utils.roi import ROIProcessor
from ppg.ppg_map import PPG_MAP
import numpy as np

def predict(video_path: str, config_path: str) -> float:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Argument parser
    # Load configuration
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)

    # Logger setup
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)


    # Load video
    logger.info("Processing the video...")
    landmarker = ROIProcessor(video_path, config)
    transformed_frames, fps = landmarker.detect_with_map()

    # Load model
    logger.info("Loading the model...")
    model_path = config["model_path"]
    model = Classifier(config["model_type"], int(config["fps_standard"]*config["seg_time_interval"]))
    model.load_model(model_path)

    predictions = []
    for segment in transformed_frames:
        logger.info("Generating PPG map...")
        ppg_map = PPG_MAP(segment, fps, config).compute_map()
        logger.info("Predicting...")
        ppg_map = torch.tensor(ppg_map, dtype=torch.float32).unsqueeze(0).to(device)
        prediction = model.predict(ppg_map, device)
        ## send to backend server ##
        predictions.append(prediction)
        
    return torch.mean(torch.stack([pred.cpu() for pred in predictions])).item()
