import yaml
import joblib
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from pycaret.classification import load_model, predict_model
from utils.logging import setup_logging
from svr.preprocess_feature import extract_feature
from data.fakeforensics import load_fakeforensics_data
# Set up logging
global logger
logger = setup_logging()

# def perform_pca(data, n_components=None):
#     """
#     Perform PCA on the given dataset.

#     Parameters:
#     - data (pd.DataFrame): Input dataset with numerical columns.
#     - n_components (int or None): Number of principal components to retain. If None, retain all components.

#     Returns:
#     - pca_df (pd.DataFrame): Transformed data with principal components.
#     - explained_variance (list): Explained variance ratio of each principal component.
#     - pca (PCA): The fitted PCA object.
#     """
#     # Standardize the data

#     # Apply PCA
#     pca = PCA(n_components=n_components)
#     principal_components = pca.fit_transform(data)

#     # Create a DataFrame for principal components
#     pca_columns = [f'PC{i+1}' for i in range(principal_components.shape[1])]
#     pca_df = pd.DataFrame(data=principal_components, columns=pca_columns)

#     # Explained variance ratio
#     explained_variance = pca.explained_variance_ratio_

#     return pca_df, explained_variance, pca

def main():
    # Load saved model and PCA
    model = load_model('best_model')
    # pca = joblib.load('pca_model.pkl')  # PCA 객체 로드

    # # argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-c', '--config_path', type=str, required=True, help="Path to the config file.")
    # args = parser.parse_args()
    config_path = '/root/25th-conference-fakebusters/model/fakecatcher/utils/config.yaml'
    # Load configuration
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)

    data = joblib.load('features_20241228_164626 copy.pkl')
    features = data['features'][:1000]
    video_labels = data['labels'][:1000]
    # csv_path = '/root/25th-conference-fakebusters/model/fakecatcher/data/test_video_list.csv'
    # video_paths, labels = load_fakeforensics_data(csv_path)
    # # Load data
    # video_paths = ['/root/data/manipulated_sequences/manipulated_sequences_2/13_14__hugging_happy__KMQ3AW6A.mp4', '/root/data/manipulated_sequences/manipulated_sequences_2/18_25__podium_speech_happy__SEGFKFJG.mp4']
    # labels = [0, 0]
    
    # logger.info(f"Loaded {len(video_paths)} videos.")
    valid_labels = []
    video_id = 0
    video_ids = []
    # Extract features
    video_features = []
    for features, label in tqdm(zip(features, video_labels), desc='Processing videos'):
        # features = extract_feature(video_path, config)
        if isinstance(features, np.ndarray):
            video_features.append(features)
            valid_labels.append(label)
            video_ids.extend([video_id] * len(features))
            video_id+=1
    # Combine features and labels
    all_features = np.vstack(video_features)
    all_labels = np.hstack([[label] * len(features) for label, features in zip(valid_labels, video_features)])
    # Normalize features
    scaler = StandardScaler()
    all_features = scaler.fit_transform(all_features)
    
    df = pd.DataFrame(all_features, columns=[f"feature_{i}" for i in range(all_features.shape[1])])
    df['target'] = all_labels  # Add the target column
    
    # all_features_pca = pca.transform(df)  # PCA 변환
    
    # pca_columns = [f"PC{i+1}" for i in range(all_features_pca.shape[1])]

    # Convert to DataFrame
    # pca_df = pd.DataFrame(all_features_pca, columns=pca_columns)
    # pca_df, _, _ = perform_pca(df)
    # expanded_labels = [label for features, label in zip(video_features, valid_labels) for _ in range(len(features))]

    # pca_df['target'] = expanded_labels  
    
    # Predict using the loaded model
    holdout_pred1 = predict_model(model, data=df)
    holdout_pred1['video_id'] = video_ids
    
    video_stats = holdout_pred1.groupby('video_id').agg(
        prediction_mean=('prediction_label', 'mean'),
        target_label=('target', 'first')  # Assume all segments in a video have the same target label
    )
    video_stats['rounded_prediction'] = video_stats['prediction_mean'].round().astype(int)
    video_stats.to_csv('result.csv', index=False)
    holdout_pred1.to_csv('features.csv')
    return video_stats

if __name__ == '__main__':
    
    print(main())