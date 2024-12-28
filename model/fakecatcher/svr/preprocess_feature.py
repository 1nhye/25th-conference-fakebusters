import yaml
import joblib
import argparse
import numpy as np
from tqdm import tqdm
from datetime import datetime
from data.fakeforensics import load_fakeforensics_data

from ppg.ppg_c import PPG_C
from ppg.ppg_g import PPG_G
from utils.roi import ROIProcessor
from utils.logging import setup_logging
from ppg.interpolate import frequency_resample
from feature.feature_extractor import FeatureExtractor


# Set up logging
global logger
logger = setup_logging()

def extract_feature(video_path, config):
    """Process a single video to extract features."""
    landmarker = ROIProcessor(video_path, config)
    R_means_array, L_means_array, M_means_array, original_fps = landmarker.detect_with_calculate()
    if R_means_array.shape[0] == 0:
        logger.warning(f"Skipping video {video_path} because R_means_array is empty.")
        return None, None

    features = []
    time_interval = config['seg_time_interval']
    target_fps = config['fps_standard']

    for i in range(R_means_array.shape[0]):
        G_R = PPG_G(R_means_array[i], original_fps).compute_signal()
        G_L = PPG_G(L_means_array[i], original_fps).compute_signal()
        G_M = PPG_G(M_means_array[i], original_fps).compute_signal()
        C_R = PPG_C(R_means_array[i], original_fps).compute_signal()
        C_L = PPG_C(L_means_array[i], original_fps).compute_signal()
        C_M = PPG_C(M_means_array[i], original_fps).compute_signal()
        
        # Segment signals
        R_ROI_G_segments = frequency_resample(G_R, time_interval, original_fps, target_fps)
        R_ROI_C_segments = frequency_resample(C_R, time_interval, original_fps, target_fps)
        L_ROI_G_segments = frequency_resample(G_L, time_interval, original_fps, target_fps)
        L_ROI_C_segments = frequency_resample(C_L, time_interval, original_fps, target_fps)
        M_ROI_G_segments = frequency_resample(G_M, time_interval, original_fps, target_fps)
        M_ROI_C_segments = frequency_resample(C_M, time_interval, original_fps, target_fps)

        # Combine segments
        ppg = [
            L_ROI_G_segments, M_ROI_G_segments, R_ROI_G_segments,
            L_ROI_C_segments, M_ROI_C_segments, R_ROI_C_segments
        ]

        # Extract features
        fe = FeatureExtractor(target_fps, *ppg)
        features.append(fe.feature_union())

    return np.array(features)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_path', type=str, required=True, help="Path to the config file.")
    args = parser.parse_args()

    # Load configuration
    with open(args.config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    feature_save_path = f"features_{current_time}.pkl"
    
    # Load data
    data_root_directory = config['data_root_directory']
    video_paths, true_labels = load_fakeforensics_data(config["meta_data_csv_path"])
    logger.info(f"Loaded {len(video_paths)} videos.")

    # Extract features
    video_features, video_labels = [], []
    for video_path, true_label in tqdm(zip(video_paths, true_labels), desc='Processing videos'):
        features = extract_feature(video_path, config)
        if features is not None:
            video_features.append(features)
            video_labels.append(true_label)
            # feature 저장장
        joblib.dump({'features': video_features, 'labels': video_labels}, feature_save_path)
    
    # feature 저장
    joblib.dump({'features': video_features, 'labels': video_labels}, feature_save_path)
    logger.info(f"Extracted features saved to {feature_save_path}.")

if __name__ == "__main__":
    main()
