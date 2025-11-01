"""
Dual-Prediction Match Analysis: Meeting Probability & Relationship Success Forecasting

Project Overview:
This machine learning pipeline implements a two-stage prediction system for dating analytics:
1. Meeting Prediction Model: Forecasts likelihood of two people meeting based on demographic and platform data
2. Relationship Success Model: Predicts relationship success probability for users who actually met

Key Features:
- Comprehensive data preprocessing with height normalization and categorical encoding
- Dual Random Forest classifiers with balanced class weights for imbalanced datasets
- Feature importance analysis for both numerical and categorical factors
- Model performance evaluation with accuracy metrics and classification reports
- Interactive visualization of factor impacts on prediction probabilities

Dataset Structure:
- Demographic features: age, height, body_type, region_origin, employment sphere
- Platform features: app_source, chat_channel, activity patterns
- Relationship metrics: sex_quality, emotional_comfort, chemistry, communication compatibility
- Target variables: met (binary), relationship_outcome (binary)

Model Architecture:
- Preprocessing: OneHotEncoding for categorical features, passthrough for numerical
- Algorithms: RandomForestClassifier with optimized hyperparameters
- Evaluation: Train-test split with stratification, comprehensive metrics

Business Applications:
- Dating platform optimization through feature importance insights
- User matching algorithm enhancement
- Relationship coaching and compatibility analysis

Author: DenisBeliaev
Date: August 2025
Version: 1.0
"""

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

# --- 4. Model Training ---

# Split data for meeting prediction model (training set only)
X_meet_train, _, y_meet_train, _ = train_test_split(X_meet, y_meet, test_size=0.15, random_state=58)
meet_pipeline.fit(X_meet_train, y_meet_train)

# Train success model only if sufficient data is available
if len(X_success) > 3:
    X_success_train, _, y_success_train, _ = train_test_split(X_success, y_success, test_size=0.15, random_state=58)
    success_pipeline.fit(X_success_train, y_success_train)
else:
    success_pipeline = None


# --- Feature Importance Analysis ---
def get_feature_importances(pipeline, model_name):
    """
    Extract feature importances from trained pipeline and create results DataFrame.

    Args:
        pipeline: Trained sklearn pipeline with preprocessor and classifier
        model_name (str): Model identifier for column naming

    Returns:
        pd.DataFrame: Feature names and their importance scores
    """
    # Get transformed feature names from preprocessing step
    feature_names_out = pipeline.named_steps['preprocessor'].get_feature_names_out()

    # Extract feature importance scores from RandomForest classifier
    importances = pipeline.named_steps['classifier'].feature_importances_

    # Create results DataFrame with feature names and importance scores
    return pd.DataFrame({
        "feature": feature_names_out,
        f"importance_{model_name}": importances
    })

# Generate feature importance table for meeting prediction model
meet_importances = get_feature_importances(meet_pipeline, "meeting")

# Check if success model was trained and merge feature importance tables
if success_pipeline:
    success_importances = get_feature_importances(success_pipeline, "success")
    # Merge importance tables by feature name, filling missing values with 0
    factors_table = pd.merge(meet_importances, success_importances, on="feature", how="outer").fillna(0)
else:
    factors_table = meet_importances

print("\nFeature Importance Table:")
# Display top 15 most important features for meeting prediction
print(factors_table.sort_values("importance_meeting", ascending=False).head(15))

# --- Model Accuracy Evaluation ---
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# For meeting prediction model - extract test data using indices
X_meet_test = meet_data.drop(X_meet_train.index)  # Get test data using remaining indices
y_meet_test = y_meet.drop(y_meet_train.index)

# Generate predictions and calculate accuracy metrics
y_meet_pred = meet_pipeline.predict(X_meet_test)
meet_accuracy = accuracy_score(y_meet_test, y_meet_pred)

print(f"\n📊 MEETING MODEL ACCURACY: {meet_accuracy:.2%}")
print("Meeting Model Classification Report:")
print(classification_report(y_meet_test, y_meet_pred))
print("Meeting Model Confusion Matrix:")
print(confusion_matrix(y_meet_test, y_meet_pred))

# For success model (if it was trained and has sufficient data)
if success_pipeline and len(X_success) > 10:
    # Extract test data using indices from training split
    X_success_test = success_data.drop(X_success_train.index)
    y_success_test = y_success.drop(y_success_train.index)

    # Generate predictions and calculate accuracy metrics
    y_success_pred = success_pipeline.predict(X_success_test)
    success_accuracy = accuracy_score(y_success_test, y_success_pred)

    print(f"\n📊 SUCCESS MODEL ACCURACY: {success_accuracy:.2%}")
    print("Success Model Classification Report:")
    print(classification_report(y_success_test, y_success_pred))


# --- Categorical Factor Impact Analysis ---
def analyze_categorical_direction(pipeline, X_data, factor, target_name):
    """
    Analyze how different categorical values affect prediction probability.

    Args:
        pipeline: Trained sklearn pipeline
        X_data: Feature dataset
        factor: Categorical column name to analyze
        target_name: Target variable name for reporting

    Returns:
        pd.DataFrame: Analysis results for each categorical value
    """
    # Calculate baseline probability
    base_prob = pipeline.predict_proba(X_data)[:, 1].mean()
    results = []

    # Analyze impact of each categorical value
    for value in X_data[factor].unique():
        X_temp = X_data.copy()
        X_temp[factor] = value
        prob = pipeline.predict_proba(X_temp)[:, 1].mean()
        direction = "↑ increases" if prob > base_prob else "↓ decreases"
        results.append((factor, value, prob, prob - base_prob, direction))

    return pd.DataFrame(results, columns=[
        "Factor", "Value", f"P({target_name}=1)", "Δ from average", "Direction"
    ])

# Analyze categorical factors for meeting prediction
cat_directions_meet = pd.concat(
    [analyze_categorical_direction(meet_pipeline, X_meet, f, "meeting") for f in cat_cols],
    ignore_index=True
)

# Sort by "Δ from average" value (descending order)
cat_directions_meet_sorted = cat_directions_meet.sort_values('Δ from average', ascending=False)

print("\nCategorical Factors (Meeting) - Sorted by Impact:")
print(cat_directions_meet_sorted.to_string(index=False))

# For relationship success analysis
if success_pipeline:
    cat_directions_success = pd.concat(
        [analyze_categorical_direction(success_pipeline, X_success, f, "success") for f in cat_cols],
        ignore_index=True
    )
    # Sort by "Δ from average" value (descending order)
    cat_directions_success_sorted = cat_directions_success.sort_values('Δ from average', ascending=False)

    print("\nCategorical Factors (Success) - Sorted by Impact:")
    print(cat_directions_success_sorted.to_string(index=False))


# --- Numerical Factor Impact Analysis ---
def plot_factor_influence(pipeline, X_data, factor, target_name):
    """
    Plot how numerical factors influence prediction probability.

    Args:
        pipeline: Trained sklearn pipeline
        X_data: Feature dataset
        factor: Numerical column name to analyze
        target_name: Target variable name for plot labels
    """
    # Create range of values from min to max of the factor
    values = np.linspace(X_data[factor].min(), X_data[factor].max(), 30)
    probs = []

    # Calculate probability for each value in the range
    for value in values:
        X_temp = X_data.copy()
        X_temp[factor] = value
        probs.append(pipeline.predict_proba(X_temp)[:, 1].mean())

    # Create influence plot
    plt.plot(values, probs, marker='o')
    plt.xlabel(factor)
    plt.ylabel(f"P({target_name}=1)")
    plt.title(f"{factor} Impact on {target_name}")
    plt.grid(True)
    plt.show()

# Analyze age and height influence on meeting probability
plot_factor_influence(meet_pipeline, X_meet, "age", "meeting")
plot_factor_influence(meet_pipeline, X_meet, "height_num", "meeting")

# Analyze age and height influence on relationship success
if success_pipeline:
    plot_factor_influence(success_pipeline, X_success, "age", "success")
    plot_factor_influence(success_pipeline, X_success, "height_num", "success")

# --- 5. Model Persistence ---
# Save trained models for future use
joblib.dump(meet_pipeline, "meet_model.pkl")
print("✓ Meeting model saved: meet_model.pkl")

if success_pipeline:
    joblib.dump(success_pipeline, "success_model.pkl")
    print("✓ Success model saved: success_model.pkl")