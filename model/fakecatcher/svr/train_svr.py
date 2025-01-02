import joblib
from tqdm import tqdm
import numpy as np
from model import Model
from utils.logging import setup_logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score  # 추가
from sklearn.metrics import mean_squared_error  # 회귀 평가를 위한 추가
from sklearn.preprocessing import StandardScaler

def main():
    data = joblib.load('features_20241228_164626.pkl')

    # Set up logging
    global logger
    logger = setup_logging()

    video_features = data['features']  # Assumed to be a list of numpy arrays
    video_labels = data['labels']     # Assumed to be a list of labels

    valid_features = []
    valid_labels = []

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

    scaler = StandardScaler()
    all_features = scaler.fit_transform(all_features)
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(all_features, all_labels, test_size=0.3, random_state=42)

    # Train SVR model
    model = Model()
    save_interval = 1000

    for i in tqdm(range(0, len(X_train), save_interval), desc="Training Progress"):
        batch_X = X_train[i:i+save_interval]
        batch_y = y_train[i:i+save_interval]
        
        # Training the model on the current batch
        model.train(batch_X, batch_y)
        
        # Saving the model after training the batch
        model.save_model(f'svr_model_{i}.pkl')
        
        # Logging the save event
        logger.info(f"Model saved at iteration {i}.")

    # Evaluate the model
    y_pred = model.predict(X_test)

    # Calculate accuracy for classification tasks
    if len(set(all_labels)) > 10:  # If the labels seem to be regression values, use RMSE
        mse = mean_squared_error(y_test, y_pred)
        logger.info(f"Mean Squared Error (MSE): {mse:.4f}")
    else:  # Assume it's a classification problem
        y_pred_rounded = np.round(y_pred).astype(int)
        accuracy = accuracy_score(y_test, y_pred_rounded)
        logger.info(f"Accuracy: {accuracy:.4f}")

if __name__ == '__main__':
    main()
