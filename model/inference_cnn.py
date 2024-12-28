from fakecatcher.cnn.classifier import Classifier
from fakecatcher.utils.roi import ROIProcessor
from fakecatcher.ppg.ppg_map import PPG_MAP
import yaml
import logging
import argparse

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config_path', type=str, required=True, help="Path to the config file.")
args = parser.parse_args()

# Load configuration
with open(args.config_path, 'r', encoding='utf-8') as file:
    config = yaml.safe_load(file)

# Logger setup
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Load model
logger.info("Loading the model...")
model_path = config["model_path"]
model = Classifier(config["model_name"])
model.load_model(model_path)


# Load video
logger.info("Processing the video...")
video_path = config["video_path"]
landmarker = ROIProcessor(video_path, config)
transformed_frames, fps = landmarker.detect_with_map()

predictions = []
for segment in transformed_frames:
    logger.info("Generating PPG map...")
    ppg_map = PPG_MAP(segment, fps, config).compute_map()
    logger.info("Predicting...")
    prediction = model.predict(ppg_map)
    ## send to backend server ##
    predictions.append(prediction)