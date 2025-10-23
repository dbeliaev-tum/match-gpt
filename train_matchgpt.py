# MatchGPT
# AI-powered insights into your dating life success
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import joblib
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# --- 1. DATA UPLOAD AND PREPROCESSING ---

# Load dataset with features for meeting prediction and relationship success
# Columns: Title, age, height, body_type, region_origin, sphere_of_employment,
#          app_source, chat_channel, met, days_to_meet, sex_quality,
#          emotional_comfort, chemistry, communication_style_match,
#          relationship_outcome, activity
data = pd.read_csv("match_gpt_testset.csv")


def process_height(height_value):
    """
    Convert height range string to numerical mean value.

    Handles height values in format '160-170' by calculating mean.
    Non-string values or values without '-' return NaN for later imputation.

    Args:
        height_value (str/any): Height value to process

    Returns:
        float: Mean height or NaN if cannot process
    """
    if isinstance(height_value, str) and '-' in height_value:
        try:
            low, high = map(int, height_value.split('-'))
            return (low + high) / 2
        except (ValueError, TypeError):
            # Handle cases where conversion to int fails
            return np.nan
    return np.nan

# Create numerical height feature from string ranges
data['height_num'] = data['height'].apply(process_height)

# Impute missing height values with median (more robust than mean for height)
height_median = data['height_num'].median()
data['height_num'] = data['height_num'].fillna(height_median)

# --- 2. DATASET PREPARATION FOR DUAL MODEL PIPELINE ---

# Create separate datasets for two predictive models:
# 1. Meeting prediction model (all users)
# 2. Relationship success model (only users who met)

# --- Meeting Prediction Dataset ---
# Target: predict whether a meeting will occur
meet_data = data.copy()

# Features for meeting prediction - demographic and platform interaction data
X_meet = meet_data[[
    'age', 'height_num', 'body_type', 'region_origin',
    'sphere_of_employment', 'app_source', 'chat_channel'
]]
y_meet = meet_data['met']  # Binary target: 1 if meeting occurred, 0 otherwise

# --- Relationship Success Dataset ---
# Target: predict relationship success (only for users who actually met)
success_data = data[data['met'] == 1].copy()

# Features for success prediction include both demographic and post-meeting metrics
X_success = success_data[[
    'age', 'height_num', 'body_type', 'region_origin',
    'sphere_of_employment', 'app_source', 'chat_channel',
    'activity', 'sex_quality', 'emotional_comfort',
    'chemistry', 'communication_style_match'
]]

# Convert relationship outcome to binary: 1 for successful (rating >= 1), 0 otherwise
# Assuming relationship_outcome is ordinal scale where positive values indicate success
y_success = success_data['relationship_outcome'].apply(lambda x: 1 if x >= 1 else 0)

# --- Categorical Feature Definition ---
# Identify categorical columns for preprocessing (one-hot encoding, etc.)
# These features require special handling in ML pipelines
cat_cols = [
    'body_type', 'region_origin', 'sphere_of_employment',
    'app_source', 'chat_channel'
]

# --- 3. PREPROCESSORS AND MODEL PIPELINES ---

"""
Define separate preprocessing and modeling pipelines for two prediction tasks:
1. Meeting prediction - uses basic demographic and platform features
2. Relationship success prediction - uses additional compatibility metrics
"""

# --- Feature Column Definitions ---
# Define feature sets to ensure consistency across pipelines
NUMERICAL_FEATURES_MEET = ['age', 'height_num']
NUMERICAL_FEATURES_SUCCESS = [
    'age', 'height_num', 'sex_quality', 'emotional_comfort',
    'chemistry', 'communication_style_match'
]

# --- Preprocessing Pipelines ---
# Meeting prediction preprocessor: handles numerical and categorical features
meet_preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', NUMERICAL_FEATURES_MEET),  # Numerical features: no transformation
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)  # Categorical: one-hot encoding
    ],
    remainder='drop',  # Explicitly drop unused columns for safety
    verbose_feature_names_out=False  # Cleaner feature names after transformation
)

# Relationship success preprocessor: includes compatibility metrics
success_preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', NUMERICAL_FEATURES_SUCCESS),  # Includes post-meeting metrics
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ],
    remainder='drop',
    verbose_feature_names_out=False
)

# --- Model Pipelines ---
# Meeting prediction pipeline with balanced class weights for imbalanced data
meet_pipeline = Pipeline([
    ('preprocessor', meet_preprocessor),
    ('classifier', RandomForestClassifier(
        n_estimators=400,        # Large ensemble for robust performance
        class_weight='balanced', # Handle class imbalance in meeting outcomes
        random_state=42,         # Reproducibility
        max_depth=7,             # Prevent overfitting, limit tree complexity
        n_jobs=-1                # Utilize all CPU cores for training
    ))
])

# Relationship success pipeline with slightly simpler trees (less data available)
success_pipeline = Pipeline([
    ('preprocessor', success_preprocessor),
    ('classifier', RandomForestClassifier(
        n_estimators=400,
        class_weight='balanced',  # Important for potentially imbalanced success outcomes
        random_state=42,
        max_depth=5,              # More restrictive due to smaller dataset size
        n_jobs=-1
    ))
])
