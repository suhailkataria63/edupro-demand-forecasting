import joblib
import os

MODELS_DIR = "models"

def save_model(model, filename):
    """Save a trained model to the models directory."""
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
    filepath = os.path.join(MODELS_DIR, filename)
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")

def load_model(filename):
    """Load a model from the models directory."""
    filepath = os.path.join(MODELS_DIR, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file {filepath} not found.")
    return joblib.load(filepath)