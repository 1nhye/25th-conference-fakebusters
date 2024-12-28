import yaml
import joblib
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pycaret.classification import load_model, predict_model
from utils.logging import setup_logging
from preprocess_feature import extract_feature

# Set up logging
global logger
logger = setup_logging()

def perform_pca(data, n_components=None):
    """
    Perform PCA on the given dataset.

    Parameters:
    - data (pd.DataFrame): Input dataset with numerical columns.
    - n_components (int or None): Number of principal components to retain. If None, retain all components.

    Returns:
    - pca_df (pd.DataFrame): Transformed data with principal components.
    - explained_variance (list): Explained variance ratio of each principal component.
    - pca (PCA): The fitted PCA object.
    """
    # Standardize the data

    # Apply PCA
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(data)

    # Create a DataFrame for principal components
    pca_columns = [f'PC{i+1}' for i in range(principal_components.shape[1])]
    pca_df = pd.DataFrame(data=principal_components, columns=pca_columns)

    # Explained variance ratio
    explained_variance = pca.explained_variance_ratio_

    return pca_df, explained_variance, pca

def main():
    # Load saved model and PCA
    model = load_model('best_model')
    pca = joblib.load('pca_model.pkl')  # PCA 객체 로드

    # argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_path', type=str, required=True, help="Path to the config file.")
    args = parser.parse_args()

    # Load configuration
    with open(args.config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)

    # Load data
    data_root_directory = config['data_root_directory']
    video_paths = ['/root/ybigta/25th-conference-fakebusters/model/fakecatcher/video/video.mp4']
    true_labels = [1]
    logger.info(f"Loaded {len(video_paths)} videos.")
    valid_labels = []

    # Extract features
    video_features = []
    for video_path, label in tqdm(zip(video_paths, true_labels), desc='Processing videos'):
        features = extract_feature(video_path, config)
        if isinstance(features, np.ndarray):
            video_features.append(features)
            valid_labels.append(label)

    # Combine features and labels
    all_features = np.vstack(video_features)
    all_labels = np.hstack([[label] * len(features) for label, features in zip(valid_labels, video_features)])
    
    # Normalize features
    scaler = StandardScaler()
    all_features = scaler.fit_transform(all_features)
    all_labels = [label] * len(features) 
    df = pd.DataFrame(all_features, columns=[f"feature_{i}" for i in range(all_features.shape[1])])
    df['target'] = all_labels  # Add the target column
    
    all_features_pca = pca.transform(df)  # PCA 변환
    pca_columns = [f"PC{i+1}" for i in range(all_features_pca.shape[1])]

    # Convert to DataFrame
    pca_df = pd.DataFrame(all_features_pca, columns=pca_columns)
    # pca_df, _, _ = perform_pca(df)
    expanded_labels = [label for features, label in zip(video_features, valid_labels) for _ in range(len(features))]

    pca_df['target'] = expanded_labels  
    
    # Predict using the loaded model
    holdout_pred1 = predict_model(model, data=pca_df)
    result =  np.mean(holdout_pred1['prediction_label'])
    return result


if __name__ == '__main__':
    print(main())
