# MatchGPT: Dual-Prediction Match Analysis System

## 📊 Project Overview

MatchGPT is a comprehensive machine learning pipeline that implements a two-stage prediction system for dating analytics:

1. **Meeting Prediction Model**: Forecasts the likelihood of two people meeting based on demographic and platform interaction data
2. **Relationship Success Model**: Predicts relationship success probability for users who actually met

This system provides data-driven insights for dating platforms, relationship coaching, and compatibility analysis.

## 🚀 Key Features

- **Dual Prediction Architecture**: Separate models for meeting probability and relationship success
- **Comprehensive Data Processing**: Advanced preprocessing with height normalization and categorical encoding
- **Balanced Classification**: Random Forest classifiers with SMOTE oversampling for imbalanced datasets
- **Feature Importance Analysis**: Identifies key factors influencing both meeting and success probabilities
- **Interactive Visualizations**: Probability curves showing how different factors impact outcomes
- **Cross-Validation**: Robust model evaluation with stratified k-fold validation

## 🏗️ Model Architecture

### Data Preprocessing
- Height range conversion to numerical values
- OneHotEncoding for categorical features
- Interaction feature creation (age × height)
- Missing value imputation

### Algorithms
- **RandomForestClassifier** with optimized hyperparameters
- **SMOTE** for handling class imbalance
- **Stratified Train-Test Split** for representative evaluation

### Feature Sets
**Meeting Prediction Features:**
- Age, Height, Body Type, Region Origin
- Employment Sphere, App Source, Chat Channel
- Age-Height Interaction

**Relationship Success Features:**
- All meeting features plus:
- Activity Level, Sex Quality, Emotional Comfort
- Chemistry, Communication Style Match

## 📁 Project Structure

```
matchgpt-project/
├── train_matchgpt.py          # Main training script
├── run_matchgpt.py            # Prediction interface
├── meet_model.pkl             # Trained meeting model
├── success_model.pkl          # Trained success model
├── match_gpt_testset.csv      # Training dataset
└── README.md                  # This file
```

## 🛠️ Installation & Usage

### Prerequisites
```bash
pip install pandas numpy scikit-learn imbalanced-learn matplotlib scipy joblib
```

### Training the Models
```bash
python train_matchgpt.py
```

This will:
- Load and preprocess the dataset
- Train both prediction models
- Generate feature importance analysis
- Create visualization plots
- Save trained models as `.pkl` files

### Making Predictions
```bash
python run_matchgpt.py
```

Edit the **USER PROFILE INPUT SECTION** in `run_matchgpt.py` to input your data:

```python
# Basic Demographics
age = 27
height = '180-190'
body_type = 'athletic'
region_origin = 'MiddleEast'

# Professional & Platform Information
sphere_employment = 'it / tech'
app_source = 'tinder'
chat_channel = 'instagram dm'
activity = 'active'

# Compatibility Metrics (Optional)
sex_quality = 3
emotional_comfort = 3
chemistry = 3
communication_style = 3
```

## 📊 Output Interpretation

### Prediction Results
```
📊 PREDICTION RESULTS:
Meeting probability: 78.5%
Relationship success probability: 89.2%
```

### Feature Importance
The system provides ranked feature importance tables showing which factors most significantly impact:
- Meeting likelihood
- Relationship success probability

### Categorical Factor Analysis
Shows how different categorical values (e.g., body types, regions, apps) affect prediction probabilities compared to average.

## 📈 Business Applications

- **Dating Platform Optimization**: Enhance matching algorithms using feature importance insights
- **User Experience Improvement**: Identify key factors that drive successful connections
- **Relationship Coaching**: Provide data-backed compatibility analysis
- **Market Research**: Understand demographic and behavioral patterns in dating

## 🔬 Technical Details

### Model Performance Metrics
- Accuracy, Precision, Recall, F1-score
- Cross-validation with 5 folds
- Stratified sampling for representative splits

### Data Requirements
- CSV format with specified column structure
- Binary target variables: `met` (0/1) and `relationship_outcome` (0/1)
- Support for mixed data types (numerical and categorical)

## 📋 Dataset Schema

Required columns in training data:
- **Demographic**: age, height, body_type, region_origin
- **Professional**: sphere_of_employment
- **Platform**: app_source, chat_channel, activity
- **Compatibility**: sex_quality, emotional_comfort, chemistry, communication_style_match
- **Targets**: met, relationship_outcome

## 🎯 Model Validation

Both models undergo comprehensive evaluation:
- Train-test split (80-20% for meeting, 83-17% for success)
- 5-fold cross-validation
- Classification reports with precision/recall metrics
- Confusion matrix analysis

## 📝 Author & Version

- **Author**: DenisBeliaev
- **Date**: August 2025
- **Version**: 1.0

## 🔮 Future Enhancements

- Improvement of user's personal life experience
- Integration with real-time dating platform APIs
- Additional compatibility metrics
- Advanced ensemble methods
- Mobile application interface
- A/B testing framework for model validation