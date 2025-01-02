import yaml
import joblib
import argparse
import numpy as np
from tqdm import tqdm
from svr.model import Model
from data.fakeforensics import load_fakeforensics_data
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
    
    return round(mean_prob)

def main():
    # path
    feature_save_path = 'features_20241228_164626 copy.pkl'
    model_saved_path = 'svr_model_15000.pkl'
    
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
    # csv_path='/root/github/25th-conference-fakebusters/model/fakecatcher/data/test_video_list.csv'
    # video_paths, true_labels = load_fakeforensics_data(csv_path)
    # logger.info(f"Loaded {len(video_paths)} videos.")
    
    data = joblib.load(feature_save_path)
    video_features = data['features']
    video_labels = data['labels']
    logger.info(f"Loaded {len(video_features)} videos.")
    valid_features = []
    valid_labels = []
    nan = 0
    for video_feature, label in tqdm(zip(video_features, video_labels), desc='Processing videos'):
        if not isinstance(video_feature, np.ndarray):
            logger.warning(f"pass video")
            nan+=1
            continue
        valid_features.append(video_feature)
        valid_labels.append(label)
    logger.warning(f"{nan/len(video_features)} 정도가 feature 추출안됨")
    
    # Evaluate model
    model = Model()
    model.load_model(model_saved_path)

    correct_predictions = 0
    for features, actual_label in zip(valid_features, valid_labels):
        features = np.array(features)
        if features.ndim == 1:
            features = features.reshape(1, -1)

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