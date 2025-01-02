import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pycaret.classification import *
from utils.logging import setup_logging

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
    data = joblib.load('features_20241228_164626.pkl')

    # Set up logging
    global logger
    logger = setup_logging()

    video_features = data['features']
    video_labels = data['labels']

    valid_features = []
    valid_labels = []

    total = len(video_features)
    not_array = 0
    # nan 제거
    nan = 0 
    nan_labels = []
    for features, label in tqdm(zip(video_features, video_labels), desc="Processing Features", total=len(video_features)):
        if not isinstance(features, np.ndarray):
            logger.warning("Features is not a numpy array. Current type: %s", type(features))
            nan += 1
            nan_labels.append(label)
            continue
        else:
            if np.isnan(features).any():
                logger.warning("Features contain NaN values. Skipping this sample.")
                nan += 1
                nan_labels.append(label)
                continue
            logger.info("Features is a numpy array and does not contain NaN. Proceeding...")
            valid_features.append(features)
            valid_labels.append(label)

    logger.warning(f"{nan/len(video_features)*100} is passed.")
    logger.warning(f"pass된 영상 중 약 {sum(nan_labels)/len(nan_labels)*100}이 label 1이다")
    
    # Combine all features and labels
    all_features = np.vstack(valid_features)
    all_labels = np.hstack([[label] * len(features) for label, features in zip(valid_labels, valid_features)])

    # Feature 정규화
    scaler = StandardScaler()
    all_features = scaler.fit_transform(all_features)
    df = pd.DataFrame(all_features, columns=[f"feature_{i}" for i in range(all_features.shape[1])])
    df['target'] = all_labels  # Add the target column
    
    # # PCA df
    # pca_df, _, pca = perform_pca(df)  # PCA 객체 반환
    # pca_df['target'] = all_labels

    # # PCA 객체 저장
    # joblib.dump(pca, 'pca_model.pkl')  # PCA 객체 저장
    # PyCaret 설정 및 모델 학습
    s1 = setup(df, target='target', session_id=123, use_gpu=True, normalize=True)
    best1 = s1.compare_models()

    # 모델 저장
    save_model(best1, 'best_model')


if __name__ == '__main__':
    main()
