import joblib
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

class Model:
    def __init__(self, kernel='rbf', C=1.0):
        """
        Initialize the SVM model with the specified parameters.

        Parameters:
        - kernel: Kernel type (e.g., 'linear', 'poly', 'rbf'). Default is 'rbf'.
        - C: Regularization parameter. Default is 1.0.
        """
        self.kernel = kernel
        self.C = C
        self.model = SVC(kernel=self.kernel, C=self.C)

    def train(self, X_train, y_train):
        """
        Train the SVM model on the training data.

        Parameters:
        - X_train: Training features.
        - y_train: Training target values.
        """
        self.model.fit(X_train, y_train)

    def predict(self, X):
        """
        Predict target values using the trained SVM model.

        Parameters:
        - X: Features for prediction.

        Returns:
        - Predicted target values.
        """
        return self.model.predict(X)

    def evaluate(self, y_true, y_pred):
        """
        Evaluate the model's performance using Accuracy.

        Parameters:
        - y_true: True target values.
        - y_pred: Predicted target values.

        Returns:
        - accuracy: Accuracy of the predictions.
        """
        accuracy = accuracy_score(y_true, y_pred)
        return accuracy

    def save_model(self, model_path):
        """Save the model to a file."""
        joblib.dump(self.model, model_path)
        print(f"Model saved to {model_path}")
    
    def load_model(self, model_path):
        """Load a saved model from a file."""
        self.model = joblib.load(model_path)
        print(f"Model loaded from {model_path}")
