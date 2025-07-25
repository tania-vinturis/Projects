"""
Machine Learning Model Training and Evaluation
Simple pipeline for training multiple models and comparing results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from data_explorer import categorize_columns
from logistic_regression import CustomLogisticRegression, MultiClassLogisticRegression
import warnings
import time
import signal

warnings.filterwarnings('ignore')


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException("Training timed out!")


def train_with_timeout(model_pipeline, X_train, y_train, timeout_seconds=300):
    """Train model with timeout protection"""

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_seconds)

    try:
        start_time = time.time()
        model_pipeline.fit(X_train, y_train)
        training_time = time.time() - start_time
        signal.alarm(0)
        print(f"Training completed in {training_time:.2f} seconds")
        return True
    except TimeoutException:
        print(f"Training timed out after {timeout_seconds} seconds!")
        return False
    except Exception as e:
        signal.alarm(0)  # Cancel timeout
        print(f"Training failed with error: {str(e)}")
        return False


def load_processed_data(train_file, test_file, target_name):
    """Load and split processed data"""
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)

    X_train = train_data.drop(columns=[target_name])
    y_train = train_data[target_name]
    X_test = test_data.drop(columns=[target_name])
    y_test = test_data[target_name]

    print(f"Loaded data: {X_train.shape[0]} train samples, {X_test.shape[0]} test samples")
    print(f"Features: {X_train.shape[1]}")
    print(f"Target distribution (train): {y_train.value_counts().sort_index().to_dict()}")
    print(f"Target distribution (test): {y_test.value_counts().sort_index().to_dict()}")

    return X_train, X_test, y_train, y_test


def create_preprocessing_pipeline(X_data):
    """Create preprocessing pipeline for features"""
    continuous_cols, categorical_cols, ordinal_cols = categorize_columns(X_data)
    all_categorical = categorical_cols + ordinal_cols

    print(f"Feature types: {len(continuous_cols)} continuous, {len(all_categorical)} categorical")

    # Check for high cardinality categorical features
    if all_categorical:
        high_cardinality_cols = []
        for col in all_categorical:
            unique_count = X_data[col].nunique()
            print(f"  {col}: {unique_count} unique values")
            if unique_count > 50:  # High cardinality
                high_cardinality_cols.append(col)

        if high_cardinality_cols:
            print(f"Warning: High cardinality features detected: {high_cardinality_cols}")
            print("These will be excluded to prevent feature explosion")
            all_categorical = [col for col in all_categorical if col not in high_cardinality_cols]

    transformers = []

    if all_categorical:
        # Limit categories
        transformers.append(('cat_encoder', OneHotEncoder(
            handle_unknown='ignore',
            sparse_output=False,
            max_categories=20,
            min_frequency=0.01
        ), all_categorical))

    if continuous_cols:
        transformers.append(('num_scaler', StandardScaler(), continuous_cols))

    if not transformers:
        from sklearn.preprocessing import FunctionTransformer
        return FunctionTransformer(lambda x: x)

    return ColumnTransformer(transformers=transformers)


def display_confusion_matrix(cm, class_names, model_name):
    """Display confusion matrix as heatmap"""
    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)

    # Set labels
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_yticklabels(class_names)

    # Add text annotations
    threshold = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = 'white' if cm[i, j] > threshold else 'black'
            ax.text(j, i, f'{cm[i, j]:d}', ha='center', va='center',
                    color=color, fontweight='bold')

    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    ax.set_title(f'{model_name} - Confusion Matrix')
    plt.tight_layout()
    plt.show()


def train_and_evaluate_models(train_path, test_path, target_col,
                              tree_params, forest_params, logistic_params, mlp_params):
    """Train multiple models and evaluate performance"""

    print(f"\nTraining models for {target_col}")
    print("=" * 50)

    # Load data
    X_train, X_test, y_train, y_test = load_processed_data(train_path, test_path, target_col)

    # Check if we need to handle the data differently
    unique_classes = sorted(y_train.unique())
    n_classes = len(unique_classes)
    is_binary = n_classes == 2

    print(f"Classification type: {'Binary' if is_binary else 'Multi-class'} ({n_classes} classes)")

    # Create preprocessing pipeline
    preprocessor = create_preprocessing_pipeline(X_train)

    # Encode target labels for sklearn models
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    class_names = [str(name) for name in label_encoder.classes_]

    # Define models based on problem type
    if is_binary:
        custom_model = CustomLogisticRegression(**logistic_params)
    else:
        custom_model = MultiClassLogisticRegression(**logistic_params)

    models = {
        'Decision Tree': DecisionTreeClassifier(**tree_params),
        'Random Forest': RandomForestClassifier(**forest_params),
        'Custom Logistic Regression': custom_model,
        'Neural Network': MLPClassifier(**mlp_params)
    }

    # Store results
    results = {}
    performance_summary = []

    # Preprocess data once for custom model
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    print(f"Preprocessed data shape: {X_train_processed.shape}")

    # Train and evaluate each model
    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")
        start_time = time.time()

        try:
            if model_name == 'Custom Logistic Regression':
                # Use original labels for custom model
                print("Training custom logistic regression...")
                model.fit(X_train_processed, y_train.values)
                y_pred = model.predict(X_test_processed)

                # Convert predictions to encoded format for metrics
                y_pred_encoded = label_encoder.transform(y_pred)
                training_time = time.time() - start_time
                print(f"Custom model training completed in {training_time:.2f} seconds")

            else:
                # Create complete pipeline for sklearn models
                full_pipeline = Pipeline([
                    ('preprocessor', preprocessor),
                    ('classifier', model)
                ])

                print(f"Starting {model_name} training (timeout: 300s)...")

                try:
                    success = train_with_timeout(full_pipeline, X_train, y_train_encoded, timeout_seconds=300)
                    if not success:
                        print(f"Skipping {model_name} due to timeout or error")
                        continue
                except:
                    print("Training without timeout ")
                    full_pipeline.fit(X_train, y_train_encoded)

                # Make predictions
                y_pred_encoded = full_pipeline.predict(X_test)
                training_time = time.time() - start_time
                print(f"{model_name} training completed in {training_time:.2f} seconds")

            # Calculate metrics
            accuracy = accuracy_score(y_test_encoded, y_pred_encoded)
            cm = confusion_matrix(y_test_encoded, y_pred_encoded)
            report = classification_report(y_test_encoded, y_pred_encoded,
                                           target_names=class_names, zero_division=0)

            # Store results
            results[model_name] = {
                'accuracy': accuracy,
                'confusion_matrix': cm,
                'classification_report': report,
                'training_time': training_time
            }

            performance_summary.append({
                'Model': model_name,
                'Accuracy': accuracy,
                'Training_Time': training_time
            })

            # Display results
            print(f"\n{model_name} Results:")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Training Time: {training_time:.2f}s")
            print("\nClassification Report:")
            print(report)

            # Show confusion matrix
            display_confusion_matrix(cm, class_names, model_name)

        except Exception as e:
            print(f"Error training {model_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            performance_summary.append({
                'Model': model_name,
                'Accuracy': 0.0,
                'Training_Time': time.time() - start_time
            })

    # Create performance comparison
    performance_df = pd.DataFrame(performance_summary).set_index('Model')
    print("\n" + "=" * 50)
    print("MODEL PERFORMANCE COMPARISON")
    print("=" * 50)
    print(performance_df.round(4))

    # Plot comparison
    plt.figure(figsize=(12, 6))
    colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold']
    bars = plt.bar(performance_df.index, performance_df['Accuracy'], color=colors)

    # Add value labels on bars
    for bar, accuracy in zip(bars, performance_df['Accuracy']):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{accuracy:.3f}', ha='center', va='bottom', fontweight='bold')

    plt.ylabel('Accuracy Score')
    plt.title('Model Performance Comparison')
    plt.ylim(0, 1.1)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()

    return results


def main():
    """Main execution function"""

    news_tree_config = {
        'max_depth': 8,
        'min_samples_leaf': 10,
        'min_samples_split': 20,
        'max_features': 'sqrt',
        'criterion': 'gini',
        'random_state': 42
    }

    news_forest_config = {
        'n_estimators': 20,
        'max_depth': 10,
        'min_samples_leaf': 5,
        'min_samples_split': 10,
        'max_features': 'sqrt',
        'criterion': 'gini',
        'bootstrap': True,
        'random_state': 42,
        'n_jobs': -1
    }

    news_logistic_config = {
        'learning_rate': 0.01,
        'max_epochs': 1000,
        'class_weights': 'balanced'
    }

    news_mlp_config = {
        'hidden_layer_sizes': (100, 50),
        'activation': 'relu',
        'solver': 'adam',
        'alpha': 0.001,
        'learning_rate_init': 0.001,
        'max_iter': 500,
        'early_stopping': True,
        'random_state': 42
    }

    heart_tree_config = {
        'max_depth': 6,
        'min_samples_leaf': 5,
        'min_samples_split': 10,
        'max_features': 'sqrt',
        'criterion': 'gini',
        'random_state': 42
    }

    heart_forest_config = {
        'n_estimators': 50,
        'max_depth': 8,
        'min_samples_leaf': 3,
        'min_samples_split': 6,
        'max_features': 'sqrt',
        'criterion': 'gini',
        'bootstrap': True,
        'random_state': 42,
        'n_jobs': -1
    }

    heart_logistic_config = {
        'learning_rate': 0.01,
        'max_epochs': 1000,
        'class_weights': {0: 1.0, 1: 3.0}
    }

    heart_mlp_config = {
        'hidden_layer_sizes': (50, 25),
        'activation': 'relu',
        'solver': 'adam',
        'alpha': 0.001,
        'learning_rate_init': 0.001,
        'max_iter': 500,
        'early_stopping': True,
        'random_state': 42
    }

    NEWS_TRAIN = '../data/news_popularity_train_processed.csv'
    NEWS_TEST = '../data/news_popularity_test_processed.csv'
    NEWS_TARGET = 'popularity_category'

    HEART_TRAIN = '../data/heart_1_train_processed.csv'
    HEART_TEST = '../data/heart_1_test_processed.csv'
    HEART_TARGET = 'chd_risk'

    print("=" * 60)
    print("NEWS POPULARITY CLASSIFICATION EXPERIMENT")
    print("=" * 60)

    try:
        news_results = train_and_evaluate_models(
            NEWS_TRAIN, NEWS_TEST, NEWS_TARGET,
            news_tree_config, news_forest_config, news_logistic_config, news_mlp_config
        )
    except Exception as e:
        print(f"Error in news experiment: {str(e)}")

    print("\n" + "=" * 60)
    print("HEART DISEASE CLASSIFICATION EXPERIMENT")
    print("=" * 60)

    try:
        heart_results = train_and_evaluate_models(
            HEART_TRAIN, HEART_TEST, HEART_TARGET,
            heart_tree_config, heart_forest_config, heart_logistic_config, heart_mlp_config
        )
    except Exception as e:
        print(f"Error in heart experiment: {str(e)}")

    print("\n" + "=" * 60)
    print("ALL EXPERIMENTS COMPLETED!")
    print("=" * 60)


if __name__ == "__main__":
    main()