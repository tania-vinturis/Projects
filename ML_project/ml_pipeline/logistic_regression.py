"""
Custom Logistic Regression Implementation
Manual implementation for binary and multi-class classification
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, ClassifierMixin


class CustomLogisticRegression(BaseEstimator, ClassifierMixin):
    """Manual logistic regression implementation"""

    def __init__(self, learning_rate=0.01, max_epochs=1000, class_weights=None, tolerance=1e-6):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.class_weights = class_weights
        self.tolerance = tolerance
        self.weights = None
        self.bias = None

        self.cost_history = []
        self.accuracy_history = []
        self.val_cost_history = []
        self.val_accuracy_history = []

    def _sigmoid(self, z):
        """Sigmoid activation function with numerical stability"""
        # Clip z to prevent overflow
        z = np.clip(z, -250, 250)
        return 1 / (1 + np.exp(-z))

    def _compute_cost(self, y_true, y_pred, sample_weights=None):
        """Compute logistic regression cost function"""
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

        if sample_weights is not None:
            cost = -np.sum(sample_weights * (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)))
            return cost / np.sum(sample_weights)
        else:
            return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def _compute_accuracy(self, y_true, y_pred, sample_weights=None):
        """Compute accuracy score"""
        predictions = (y_pred >= 0.5).astype(int)
        correct = (predictions == y_true).astype(float)

        if sample_weights is not None:
            return np.sum(sample_weights * correct) / np.sum(sample_weights)
        else:
            return np.mean(correct)

    def _get_sample_weights(self, y):
        """Compute sample weights for class balancing"""
        if self.class_weights is None:
            return None

        if self.class_weights == 'balanced':
            # Compute balanced weights
            unique_classes, counts = np.unique(y, return_counts=True)
            total_samples = len(y)
            n_classes = len(unique_classes)

            class_weight_dict = {}
            for cls, count in zip(unique_classes, counts):
                class_weight_dict[cls] = total_samples / (n_classes * count)

            weights = np.array([class_weight_dict[label] for label in y])
            return weights.astype(float)

        # Manual weights provided
        weights = np.array([self.class_weights.get(label, 1.0) for label in y])
        return weights.astype(float)

    def fit(self, X, y, X_val=None, y_val=None):
        """Train the logistic regression model"""
        # Clear previous history
        self.cost_history = []
        self.accuracy_history = []
        self.val_cost_history = []
        self.val_accuracy_history = []

        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)

        # Handle multi-class case by converting to binary for now
        if len(np.unique(y)) > 2:
            print("Warning: Multi-class detected. Converting to binary (class 0 vs rest)")
            y = (y != 0).astype(int)

        n_samples, n_features = X.shape

        self.weights = np.random.normal(0, 0.01, n_features)
        self.bias = 0.0

        sample_weights = self._get_sample_weights(y)

        if X_val is not None and y_val is not None:
            X_val = np.array(X_val)
            y_val = np.array(y_val)
            if len(np.unique(y_val)) > 2:
                y_val = (y_val != 0).astype(int)
            val_sample_weights = self._get_sample_weights(y_val)

        prev_cost = float('inf')

        for epoch in range(self.max_epochs):
            # Forward pass
            z = np.dot(X, self.weights) + self.bias
            predictions = self._sigmoid(z)

            # Compute cost and accuracy
            cost = self._compute_cost(y, predictions, sample_weights)
            accuracy = self._compute_accuracy(y, predictions, sample_weights)

            self.cost_history.append(cost)
            self.accuracy_history.append(accuracy)

            # Compute gradients
            dz = predictions - y

            if sample_weights is not None:
                dw = np.dot(X.T, dz * sample_weights) / np.sum(sample_weights)
                db = np.sum(dz * sample_weights) / np.sum(sample_weights)
            else:
                dw = np.dot(X.T, dz) / n_samples
                db = np.sum(dz) / n_samples

            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # Validation evaluation
            if X_val is not None and y_val is not None:
                z_val = np.dot(X_val, self.weights) + self.bias
                val_predictions = self._sigmoid(z_val)
                val_cost = self._compute_cost(y_val, val_predictions, val_sample_weights)
                val_accuracy = self._compute_accuracy(y_val, val_predictions, val_sample_weights)

                self.val_cost_history.append(val_cost)
                self.val_accuracy_history.append(val_accuracy)

            # Check for convergence
            if abs(prev_cost - cost) < self.tolerance:
                print(f"Converged at epoch {epoch + 1}")
                break

            prev_cost = cost

            # Print progress
            if (epoch + 1) % 200 == 0:
                print(f"Epoch {epoch + 1}/{self.max_epochs}, Cost: {cost:.4f}, Accuracy: {accuracy:.4f}")

        return self

    def predict_proba(self, X):
        """Predict class probabilities"""
        if self.weights is None:
            raise ValueError("Model must be fitted before making predictions")

        X = np.array(X)
        z = np.dot(X, self.weights) + self.bias
        probabilities = self._sigmoid(z)

        return np.column_stack([1 - probabilities, probabilities])

    def predict(self, X):
        """Predict class labels"""
        if self.weights is None:
            raise ValueError("Model must be fitted before making predictions")

        X = np.array(X)
        z = np.dot(X, self.weights) + self.bias
        probabilities = self._sigmoid(z)
        return (probabilities >= 0.5).astype(int)

    def score(self, X, y):
        """Compute accuracy score"""
        predictions = self.predict(X)
        return np.mean(predictions == y)

    def plot_training_history(self):
        """Plot training history"""
        if not self.cost_history:
            print("No training history available")
            return

        epochs = range(len(self.cost_history))

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # Accuracy plot
        ax1.plot(epochs, self.accuracy_history, 'b-', label='Training', linewidth=2)
        if self.val_accuracy_history:
            ax1.plot(epochs, self.val_accuracy_history, 'r-', label='Validation', linewidth=2)
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Training Progress')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Cost plot
        ax2.plot(epochs, self.cost_history, 'b-', label='Training', linewidth=2)
        if self.val_cost_history:
            ax2.plot(epochs, self.val_cost_history, 'r-', label='Validation', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Cost')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


class MultiClassLogisticRegression(BaseEstimator, ClassifierMixin):
    """Multi-class logistic regression using one-vs-rest strategy"""

    def __init__(self, learning_rate=0.01, max_epochs=1000, class_weights=None):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.class_weights = class_weights
        self.classifiers = {}
        self.classes = None

    def fit(self, X, y):
        """Train one-vs-rest classifiers"""
        X = np.array(X)
        y = np.array(y)

        self.classes = np.unique(y)

        for class_label in self.classes:
            print(f"Training classifier for class {class_label}")

            # Create binary labels
            binary_y = (y == class_label).astype(int)

            # Train binary classifier
            classifier = CustomLogisticRegression(
                learning_rate=self.learning_rate,
                max_epochs=self.max_epochs,
                class_weights=self.class_weights
            )

            classifier.fit(X, binary_y)
            self.classifiers[class_label] = classifier

        return self

    def predict_proba(self, X):
        """Predict class probabilities"""
        X = np.array(X)
        n_samples = X.shape[0]
        n_classes = len(self.classes)

        # Get probabilities
        probabilities = np.zeros((n_samples, n_classes))

        for i, class_label in enumerate(self.classes):
            classifier = self.classifiers[class_label]
            # Get probability of positive class
            class_probs = classifier.predict_proba(X)[:, 1]
            probabilities[:, i] = class_probs

        # Normalize probabilities
        row_sums = probabilities.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        probabilities = probabilities / row_sums

        return probabilities

    def predict(self, X):
        """Predict class labels"""
        probabilities = self.predict_proba(X)
        predicted_indices = np.argmax(probabilities, axis=1)
        return self.classes[predicted_indices]

    def score(self, X, y):
        """Compute accuracy score"""
        predictions = self.predict(X)
        return np.mean(predictions == y)