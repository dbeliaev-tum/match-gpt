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
- Cross-validation for robust model evaluation
- Interactive visualization of factor impacts on prediction probabilities

Dataset Structure:
- Demographic features: age, height, body_type, region_origin, employment sphere
- Platform features: app_source, chat_channel, activity patterns
- Relationship metrics: sex_quality, emotional_comfort, chemistry, communication compatibility
- Target variables: met (binary), relationship_outcome (binary)

Model Architecture:
- Preprocessing: OneHotEncoding for categorical features, passthrough for numerical
- Algorithms: RandomForestClassifier with optimized hyperparameters
- Evaluation: Cross-validation and train-test split with stratification

Business Applications:
- Dating platform optimization through feature importance insights
- User matching algorithm enhancement
- Relationship coaching and compatibility analysis

Author: DenisBeliaev
Date: August 2025
Version: 3.0
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
from scipy.interpolate import make_interp_spline
import matplotlib.pyplot as plt
import joblib
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# --- 1. DATA UPLOAD AND PREPROCESSING ---

# Load dataset with features for meeting prediction and relationship success
data = pd.read_csv("match_gpt_testset.csv")

def process_height(height_value):
    """Convert height ranges to numerical values and handle missing data"""
    if isinstance(height_value, str) and '-' in height_value:
        try:
            low, high = map(int, height_value.split('-'))
            return (low + high) / 2
        except (ValueError, TypeError):
            return np.nan
    elif isinstance(height_value, (int, float)) and 100 <= height_value <= 250:
        return height_value
    return np.nan

# Create numerical height feature from string ranges
data['height_num'] = data['height'].apply(process_height)

# Impute missing height values with median
height_median = data['height_num'].median()
data['height_num'] = data['height_num'].fillna(height_median)

# --- 2. DATASET PREPARATION FOR DUAL MODEL PIPELINE ---

# Create separate datasets for two predictive models
meet_data = data.copy()

# Create interaction feature for enhanced predictive power
data['age_height_interaction'] = data['age'] * data['height_num']
meet_data['age_height_interaction'] = meet_data['age'] * meet_data['height_num']

# --- Meeting Prediction Dataset ---
X_meet = meet_data[[
    'age', 'height_num', 'age_height_interaction', 'body_type', 'region_origin',
    'sphere_of_employment', 'app_source', 'chat_channel'
]]
y_meet = meet_data['met']

# --- Relationship Success Dataset ---
success_data = data[data['met'] == 1].copy()
if len(success_data) > 0:
    success_data['age_height_interaction'] = success_data['age'] * success_data['height_num']
    X_success = success_data[[
        'age', 'height_num', 'age_height_interaction', 'body_type', 'region_origin',
        'sphere_of_employment', 'app_source', 'chat_channel',
        'activity', 'sex_quality', 'emotional_comfort',
        'chemistry', 'communication_style_match'
    ]]
    y_success = success_data['relationship_outcome'].apply(lambda x: 1 if x >= 1 else 0)

# --- Feature Column Definitions ---
cat_cols_meet = [
    'body_type', 'region_origin', 'sphere_of_employment',
    'app_source', 'chat_channel'
]

cat_cols_success = [
    'body_type', 'region_origin', 'sphere_of_employment',
    'app_source', 'chat_channel'
    #, 'activity'
]

NUMERICAL_FEATURES_MEET = ['age', 'height_num', 'age_height_interaction']
NUMERICAL_FEATURES_SUCCESS = [
    'age', 'height_num', 'age_height_interaction', 'sex_quality', 'emotional_comfort',
    'chemistry', 'communication_style_match'
]

# --- 3. PREPROCESSORS AND MODEL PIPELINES ---

# Meeting prediction preprocessor
meet_preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', NUMERICAL_FEATURES_MEET),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols_meet)
    ],
    remainder='drop',
    verbose_feature_names_out=False
)

# Relationship success preprocessor
success_preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', NUMERICAL_FEATURES_SUCCESS),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols_success)
    ],
    remainder='drop',
    verbose_feature_names_out=False
)

# --- Model Pipelines ---
meet_pipeline = ImbPipeline([
    ('preprocessor', meet_preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(
        n_estimators=500,
        class_weight='balanced',
        random_state=42,
        max_depth=8,
        min_samples_split=10,
        n_jobs=-1
    ))
])

success_pipeline = ImbPipeline([
    ('preprocessor', success_preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(
        n_estimators=500,
        max_depth=9,
        min_samples_split=7,
        min_samples_leaf=2,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    ))
])

# --- 4. MODEL TRAINING AND EVALUATION ---

# Split data for meeting prediction model
meet_indices = X_meet.index
X_meet_train, X_meet_test, y_meet_train, y_meet_test = train_test_split(
    X_meet, y_meet, test_size=0.2, random_state=58, stratify=y_meet
)

test_indices = X_meet_test.index
X_meet_test_data = meet_data.loc[test_indices]

# Train meeting model
meet_pipeline.fit(X_meet_train, y_meet_train)

# Cross-validation for meeting model
from sklearn.model_selection import cross_validate

cv_results_meet = cross_validate(
    meet_pipeline, X_meet, y_meet,
    cv=5, scoring=['accuracy', 'precision', 'recall', 'f1'],
    return_train_score=True
)

# Train success model only if sufficient data is available
if len(X_success) > 3:
    X_success_train, X_success_test, y_success_train, y_success_test = train_test_split(
        X_success, y_success, test_size=0.17, random_state=58, stratify=y_success
    )
    success_pipeline.fit(X_success_train, y_success_train)

    # Cross-validation for success model
    cv_results_success = cross_validate(
        success_pipeline, X_success, y_success,
        cv=5, scoring=['accuracy', 'precision', 'recall', 'f1'],
        return_train_score=True
    )
else:
    success_pipeline = None
    X_success_test = None
    y_success_test = None

# --- 5. MODEL EVALUATION AND REPORTING ---

# --- Feature Importance Analysis ---
def get_feature_importances(pipeline, model_name):
    """Extract feature importances from trained pipeline and create results DataFrame"""
    feature_names_out = pipeline.named_steps['preprocessor'].get_feature_names_out()
    importances = pipeline.named_steps['classifier'].feature_importances_

    return pd.DataFrame({
        "feature": feature_names_out,
        f"importance_{model_name}": importances
    })

# Generate feature importance tables
meet_importances = get_feature_importances(meet_pipeline, "meeting")

if success_pipeline:
    success_importances = get_feature_importances(success_pipeline, "success")
    factors_table = pd.merge(meet_importances, success_importances, on="feature", how="outer").fillna(0)
else:
    factors_table = meet_importances

print("\nMEET FEATURE IMPORTANCE TABLE")
meet_sorted = factors_table[['feature', 'importance_meeting']].sort_values('importance_meeting', ascending=False).head(14)
print(meet_sorted.to_string(index=False))

# --- Model Performance Reporting ---
y_meet_pred = meet_pipeline.predict(X_meet_test)
meet_accuracy = accuracy_score(y_meet_test, y_meet_pred)

print(f"\n📊 MEETING MODEL ACCURACY: {meet_accuracy:.2%}")
print("Meeting Model Classification Report:")
report_dict_meet = classification_report(y_meet_test, y_meet_pred, digits=2, output_dict=True)
for key in ['0', '1', 'macro avg', 'weighted avg']:
    if 'support' in report_dict_meet[key]:
        del report_dict_meet[key]['support']
df_report = pd.DataFrame(report_dict_meet).T
if 'accuracy' in df_report.index:
    df_report = df_report.drop('accuracy')
print(df_report.to_string(float_format="%.2f"))

print("\n📊 Cross-Validation Results (Meeting Model):")
for metric in ['accuracy', 'precision', 'recall', 'f1']:
    print(f"  {metric.capitalize()}: {cv_results_meet[f'test_{metric}'].mean():.3f} (±{cv_results_meet[f'test_{metric}'].std():.3f})")

if success_pipeline:
    print("\nSUCCESS FEATURE IMPORTANCE TABLE")
    success_sorted = factors_table[['feature', 'importance_success']].sort_values('importance_success', ascending=False).head(14)
    print(success_sorted.to_string(index=False))

if success_pipeline and X_success_test is not None:
    y_success_pred = success_pipeline.predict(X_success_test)
    success_accuracy = accuracy_score(y_success_test, y_success_pred)

    print(f"\n📊 SUCCESS MODEL ACCURACY: {success_accuracy:.2%}")
    print("Success Model Classification Report:")
    report_dict_success = classification_report(y_success_test, y_success_pred, digits=2, output_dict=True)
    for key in ['0', '1', 'macro avg', 'weighted avg']:
        if 'support' in report_dict_success[key]:
            del report_dict_success[key]['support']
    df_report = pd.DataFrame(report_dict_success).T
    if 'accuracy' in df_report.index:
        df_report = df_report.drop('accuracy')
    print(df_report.to_string(float_format="%.2f"))

    print("\n📊 Cross-Validation Results (Success Model):")
    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        print(
            f"  {metric.capitalize()}: {cv_results_success[f'test_{metric}'].mean():.3f} (±{cv_results_success[f'test_{metric}'].std():.3f})")

# --- Categorical Factor Impact Analysis ---
def analyze_categorical_direction(pipeline, X_data, factor, target_name):
    """Analyze how different categorical values affect prediction probability"""
    base_prob = pipeline.predict_proba(X_data)[:, 1].mean()
    results = []

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
    [analyze_categorical_direction(meet_pipeline, X_meet, f, "meeting") for f in cat_cols_meet],
    ignore_index=True
)

cat_directions_meet_sorted = cat_directions_meet.sort_values('Δ from average', ascending=False)

print("\nCategorical Factors (Meeting) - Sorted by Impact:")
print(cat_directions_meet_sorted.to_string(index=False))

# For relationship success analysis
if success_pipeline:
    cat_directions_success = pd.concat(
        [analyze_categorical_direction(success_pipeline, X_success, f, "success") for f in cat_cols_success],
        ignore_index=True
    )
    cat_directions_success_sorted = cat_directions_success.sort_values('Δ from average', ascending=False)

    print("\nCategorical Factors (Success) - Sorted by Impact:")
    print(cat_directions_success_sorted.to_string(index=False))

# --- Numerical Factor Impact Analysis ---
def plot_factor_influence(pipeline, X_data, factor, target_name, y_step=None):
    """
    Visualize the relationship between numerical features and prediction probabilities.

    Creates smoothed probability curves using spline interpolation for better
    visualization of feature impact trends.

    Args:
        pipeline: Trained sklearn pipeline with preprocessor and classifier
        X_data: Feature dataset for analysis
        factor: Numerical feature name to analyze
        target_name: Target variable name for plot labeling
        y_step: Optional step size for Y-axis ticks to control granularity
    """
    # Generate base range of values for probability calculation
    values = np.linspace(X_data[factor].min(), X_data[factor].max(), 30)
    probs = []

    # Calculate prediction probabilities across the feature value range
    for value in values:
        X_temp = X_data.copy()
        X_temp[factor] = value
        probs.append(pipeline.predict_proba(X_temp)[:, 1].mean())

    # Apply spline interpolation for smooth curve visualization
    x_smooth = np.linspace(values.min(), values.max(), 200)
    spline = make_interp_spline(values, probs)
    y_smooth = spline(x_smooth)

    # Initialize plot with optimized dimensions
    plt.figure(figsize=(6, 4))

    # Plot smoothed probability curve with mint color scheme
    plt.plot(x_smooth, y_smooth, color="#3EB489", linewidth=2.5)

    # Configure axis labels and title
    plt.xlabel(factor, fontsize=11)
    plt.ylabel(f"P({target_name}=1)", fontsize=11)
    plt.title(f"{factor} Impact on {target_name}", fontsize=12, pad=10)

    # Add subtle grid for better readability
    plt.grid(alpha=0.3)

    # Apply custom Y-axis tick spacing if specified
    if y_step is not None:
        ymin, ymax = min(y_smooth), max(y_smooth)
        plt.yticks(np.arange(round(ymin, 3), round(ymax + y_step, 3), y_step))

    # Optimize layout and render plot
    plt.tight_layout()
    plt.show()

# --- Execute Factor Influence Analysis ---

# Analyze feature impacts on meeting probability
plot_factor_influence(meet_pipeline, X_meet, "age", "meeting", y_step=0.065)
plot_factor_influence(meet_pipeline, X_meet, "height_num", "meeting", y_step=0.03)

# Analyze feature impacts on relationship success probability
if success_pipeline:
    plot_factor_influence(success_pipeline, X_success, "age", "success", y_step=0.005)
    plot_factor_influence(success_pipeline, X_success, "height_num", "success", y_step=0.0025)


# --- 6. MODEL PERSISTENCE ---
joblib.dump(meet_pipeline, "meet_model.pkl")
print("✓ Meeting model saved: meet_model.pkl")

if success_pipeline:
    joblib.dump(success_pipeline, "success_model.pkl")
    print("✓ Success model saved: success_model.pkl")