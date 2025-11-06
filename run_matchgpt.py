import pandas as pd
import joblib

# Load trained models
try:
    meet_pipeline = joblib.load("meet_model.pkl")
    print("✓ Meeting model loaded: meet_model.pkl")
except FileNotFoundError:
    meet_pipeline = None
    print("⚠ Meet model not found")

try:
    success_pipeline = joblib.load("success_model.pkl")
    print("✓ Success model loaded: success_model.pkl")
except FileNotFoundError:
    success_pipeline = None
    print("⚠ Success model not found")