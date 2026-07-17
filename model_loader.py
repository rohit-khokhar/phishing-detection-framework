import os
import joblib

DEPLOY_DIR = "deploy_artifacts"

def load_artifact(filename):
    path = os.path.join(DEPLOY_DIR, filename)

    if not os.path.exists(path):
        raise FileNotFoundError(f"{filename} not found")

    return joblib.load(path)