import yaml
import joblib
import argparse
import numpy as np
from tqdm import tqdm
from svr.model import Model
from data.fakeavceleb import load_data
from utils.logging import setup_logging
from preprocess_feature import extract_feature

# Set up logging
global logger
logger = setup_logging()

def majority_voting(probabilities):
    """
    Perform majority voting on the predicted probabilities.
    """
    mean_prob = np.mean(probabilities)
    return mean_prob

def main():
    # path
    feature_save_path = 'inference_feature.pkl'
    model_saved_path = 'svr_model_2900.pkl'
    
    # argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_path', type=str, required=True, help="Path to the config file.")
    args = parser.parse_args()

    # Load configuration
    with open(args.config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)

    valid_features = []
    valid_labels = []

    # Load data
    data_root_directory = config['data_root_directory']
    video_paths, true_labels = load_data(data_root_directory, config["meta_data_csv_path"])
    logger.info(f"Loaded {len(video_paths)} videos.")

   
    # Extract features
    video_features, video_labels = [], []
    for video_path, true_label in tqdm(zip(video_paths, true_labels), desc='Processing videos'):
        features = extract_feature(video_path, config)
        if features is not None:
            video_features.append(features)
            video_labels.append(true_label)
            # feature 저장
        joblib.dump({'features': video_features, 'labels': video_labels}, feature_save_path)
    

    # Evaluate model
    model = Model()
    model.load_model(model_saved_path)

    correct_predictions = 0
    for features, actual_label in zip(video_features, video_labels):
        if not isinstance(features, np.ndarray):
            continue
        segment_probabilities = model.predict(features)
        predicted_label = majority_voting(segment_probabilities)
        logger.info(f"Actual: {actual_label}, Predicted: {predicted_label}")

        if predicted_label == actual_label:
            correct_predictions += 1

    accuracy = correct_predictions / len(video_features)
    logger.info(f"Video-level Accuracy: {accuracy * 100:.2f}%")
    print(f"Video-level Accuracy: {accuracy * 100:.2f}%")

if __name__=='__main__':
    main()