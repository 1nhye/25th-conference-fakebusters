import yaml
import logging
import argparse
from tqdm import tqdm
from utils.roi import ROIProcessor
from ppg.ppg_map import PPG_MAP
from ppg.ppg_c import PPG_C

from data.fakeforensics import load_fakeforensics_data
import json

# Argument parser
#parser = argparse.ArgumentParser()
# parser.add_argument('-c', '--config_path', type=str, required=True, help="Path to the config file.")
config_path = '/Users/treecollector/Desktop/test/model/fakecatcher/utils/config.yaml'
#args = parser.parse_args()

# # Load configuration
with open(config_path, 'r', encoding='utf-8') as file:
    config = yaml.safe_load(file)

# # Logger setup
# console_handler = logging.StreamHandler()
# console_handler.setLevel(logging.INFO)
# console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

# file_handler = logging.FileHandler('app.log', mode='w')
# file_handler.setLevel(logging.DEBUG)
# file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

logger = logging.getLogger()
# logger.setLevel(logging.DEBUG)
# logger.addHandler(console_handler)
# logger.addHandler(file_handler)

# Paths
root_directory = '/root/deepfake_detection'
model_path = config["model_path"]
meta_data_csv_path = config["meta_data_csv_path"]

# Load data
logger.info("Loading video paths and labels...")
video_paths, true_labels = load_fakeforensics_data(meta_data_csv_path)

# Process videos
logger.info("Processing videos to generate PPG maps...")
results = []
for i, (video_path, true_label) in enumerate(tqdm(zip(video_paths, true_labels), total=len(video_paths), desc="Processing videos")):

#cv2.imwrite("/Users/treecollector/Desktop/test/model/fakecatcher/ppg/ppg_map.png", ppg_map)

    # Process video with ROIProcessor
    landmarker = ROIProcessor(video_path="/Volumes/Hoesu_ssd/ff_data/original_sequences/actors/01__outside_talking_still_laughing.mp4", config=config)
    transformed_frames, fps = landmarker.detect_with_map()
    for segment in transformed_frames:
        # Compute PPG Map
        ppg_map = PPG_MAP(segment, fps, config).compute_map()
        breakpoint()
        print("shape of ppg_map: ", ppg_map.shape)
        # Append results
        results.append({
            'label': true_label,
            'ppg_map': ppg_map.tolist()
        })
        # Save results to JSON file

        output_json_path = "ppg_map_results.json"
        with open(output_json_path, 'w', encoding='utf-8') as json_file:
            json.dump(results, json_file, ensure_ascii=False, indent=4)

        logger.info(f"Results saved to {output_json_path}")