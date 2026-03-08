import joblib
import json
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


def save_metadata(metadata, filename="prediction_metadata.json"):
    """Save JSON metadata to the models directory."""
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
    filepath = os.path.join(MODELS_DIR, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
    print(f"Metadata saved to {filepath}")


def load_metadata(filename="prediction_metadata.json"):
    """Load JSON metadata from the models directory."""
    filepath = os.path.join(MODELS_DIR, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Metadata file {filepath} not found.")
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)
