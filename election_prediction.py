# -*- coding: utf-8 -*-
"""
This script implements a complete pipeline for a neural network-based election prediction task.
It covers the following steps:
1. Data generation (`dataIn.txt`, `dataOut.txt`).
2. Data loading and preprocessing with orientation checks.
3. Splitting data into training and testing sets.
4. Building a single-hidden-layer neural network with 'logsig' (sigmoid) activation.
5. Training the model with EarlyStopping.
6. Evaluating the model's performance using various metrics (accuracy, confusion matrix, ROC-AUC).
7. Visualizing the results (learning curves, confusion matrix, PCA projection).
8. Comparing the neural network's performance with classical ML models (Logistic Regression, Random Forest).
9. Saving the trained model.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
)
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from tensorflow import keras
from keras._tf_keras.keras import layers
from keras._tf_keras.keras.callbacks import EarlyStopping
import os

# --- Configuration ---
N_SAMPLES = 200  # Number of voters
INPUT_FEATURES = 12
TEST_SIZE = 0.3
RANDOM_STATE = 42
HIDDEN_UNITS = 16
EPOCHS = 100
BATCH_SIZE = 16
PATIENCE = 10
MODEL_FILENAME = "mlp_elections_model.h5"
PLOTS_DIR = "plots"


def generate_data(n_samples=N_SAMPLES, input_features=INPUT_FEATURES):
    """
    Generates and saves synthetic election data.
    - dataIn.txt: (input_features x n_samples) voter answers.
    - dataOut.txt: (2 x n_samples) one-hot encoded results.
    """
    print("--- 1. Generating Data ---")
    np.random.seed(RANDOM_STATE)

    # Generate features (12xN)
    X = np.random.randint(0, 2, size=(input_features, n_samples))

    # Generate labels (one-hot 2xN)
    labels = np.random.randint(0, 2, size=(n_samples,))
    Y = np.zeros((2, n_samples), dtype=int)
    Y[labels, np.arange(n_samples)] = 1

    np.savetxt("dataIn.txt", X, fmt="%d")
    np.savetxt("dataOut.txt", Y, fmt="%d")
    print(f"Generated dataIn.txt with shape: {X.shape}")
    print(f"Generated dataOut.txt with shape: {Y.shape}")
    print("-" * 25)
    return X, Y


def load_and_preprocess_data():
    """
    Loads data from files, checks orientation, and preprocesses it.
    Returns X (N, 12) and Y (N, 2).
    """
    print("--- 2. Loading and Preprocessing Data ---")
    X_raw = np.loadtxt("dataIn.txt", dtype=int)
    Y_raw = np.loadtxt("dataOut.txt", dtype=int)

    # --- Transpose if necessary to get (N, features) format ---
    # Handle features
    if X_raw.shape[0] == INPUT_FEATURES and X_raw.shape[1] != INPUT_FEATURES:
        X = X_raw.T
    else:
        X = X_raw

    # Handle labels
    if Y_raw.shape[0] == 2 and Y_raw.shape[1] != 2:
        Y = Y_raw.T
    else:
        Y = Y_raw

    # Convert to float32 for TensorFlow
    X = X.astype("float32")
    Y = Y.astype("float32")

    print(f"Loaded X with final shape (N x features): {X.shape}")
    print(f"Loaded Y with final shape (N x classes): {Y.shape}")
    print("-" * 25)
    return X, Y


def build_nn_model(input_dim, hidden_units):
    """
    Builds the Keras Sequential model with one hidden 'logsig' layer.
    """
    print("--- 3. Building Neural Network Model ---")
    model = keras.Sequential(
        [
            layers.Input(shape=(input_dim,)),
            layers.Dense(
                hidden_units, activation="sigmoid", name="hidden_logsig_layer"
            ),
            layers.Dense(2, activation="softmax", name="output_layer"),
        ]
    )
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    model.summary()
    print("-" * 25)
    return model


def train_model(model, X_train, y_train, X_test, y_test):
    """
    Trains the model with EarlyStopping callback.
    """
    print("--- 4. Training Model ---")
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=PATIENCE, restore_best_weights=True
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping],
        verbose=1,
    )
    print("Training complete.")
    print("-" * 25)
    return history


def evaluate_and_report(model, X_test, y_test):
    """
    Evaluates the model and prints a comprehensive report.
    """
    print("--- 5. Evaluating Model Performance ---")
    y_prob = model.predict(X_test)
    y_pred = np.argmax(y_prob, axis=1)
    y_true = np.argmax(y_test, axis=1)

    # Accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {accuracy:.4f}\n")

    # Classification Report
    print("Classification Report:")
    print(
        classification_report(
            y_true, y_pred, target_names=["Ruling Party", "Opposition"]
        )
    )

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:\n", cm)

    # ROC-AUC Score
    # For "Opposition" class (index 1)
    fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1], pos_label=1)
    roc_auc = auc(fpr, tpr)
    print(f"\nROC AUC (for Opposition class): {roc_auc:.4f}")
    print("-" * 25)
    return cm, y_true, y_pred


def visualize_results(history, cm, X_test, y_true):
    """
    Generates and saves all required plots.
    """
    print("--- 6. Visualizing Results ---")
    if not os.path.exists(PLOTS_DIR):
        os.makedirs(PLOTS_DIR)

    # 1. Loss and Accuracy curves
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title("Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(os.path.join(PLOTS_DIR, "learning_curves.png"))
    plt.show()

    # 2. Confusion Matrix Heatmap
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Ruling Party", "Opposition"],
        yticklabels=["Ruling Party", "Opposition"],
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.savefig(os.path.join(PLOTS_DIR, "confusion_matrix.png"))
    plt.show()

    # 3. PCA Projection
    pca = PCA(n_components=2)
    X_test_pca = pca.fit_transform(X_test)

    plt.figure(figsize=(8, 7))
    scatter = plt.scatter(
        X_test_pca[:, 0], X_test_pca[:, 1], c=y_true, cmap="coolwarm", alpha=0.8
    )
    plt.title("PCA Projection of Test Data (Colored by True Labels)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(handles=scatter.legend_elements()[0], labels=["Ruling Party", "Opposition"])
    plt.savefig(os.path.join(PLOTS_DIR, "pca_projection.png"))
    plt.show()
    print(f"Plots saved to '{PLOTS_DIR}' directory.")
    print("-" * 25)

def compare_with_classical_models(X_train, y_train, X_test, y_test):
    """
    Trains and evaluates Logistic Regression and Random Forest models.
    """
    print("--- 7. Comparing with Classical Models ---")
    y_train_labels = np.argmax(y_train, axis=1)
    y_test_labels = np.argmax(y_test, axis=1)

    # Logistic Regression
    lr_model = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
    lr_model.fit(X_train, y_train_labels)
    y_pred_lr = lr_model.predict(X_test)
    lr_accuracy = accuracy_score(y_test_labels, y_pred_lr)
    print(f"Logistic Regression Accuracy: {lr_accuracy:.4f}")

    # Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    rf_model.fit(X_train, y_train_labels)
    y_pred_rf = rf_model.predict(X_test)
    rf_accuracy = accuracy_score(y_test_labels, y_pred_rf)
    print(f"Random Forest Accuracy: {rf_accuracy:.4f}")
    print("-" * 25)


def main():
    """
    Main function to run the entire pipeline.
    """
    # Step 1: Generate data (or use existing files)
    if not (os.path.exists("dataIn.txt") and os.path.exists("dataOut.txt")):
        generate_data()

    # Step 2: Load and preprocess data
    X, Y = load_and_preprocess_data()

    # Step 3: Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=np.argmax(Y, axis=1)
    )
    print(f"Data split into: Train {X_train.shape}, Test {X_test.shape}")
    print("-" * 25)

    # Step 4: Build and train the neural network
    model = build_nn_model(input_dim=X.shape[1], hidden_units=HIDDEN_UNITS)
    history = train_model(model, X_train, y_train, X_test, y_test)

    # Step 5: Evaluate the model
    cm, y_true, _ = evaluate_and_report(model, X_test, y_test)

    # Step 6: Visualize results
    visualize_results(history, cm, X_test, y_true)

    # Step 7: Compare with other models
    compare_with_classical_models(X_train, y_train, X_test, y_test)

    # Step 8: Save the final model
    print("--- 8. Saving Model ---")
    model.save(MODEL_FILENAME)
    print(f"Model saved to {MODEL_FILENAME}")
    print("-" * 25)

    print("Pipeline finished successfully!")


if __name__ == "__main__":
    main()