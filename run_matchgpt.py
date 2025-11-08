"""
MatchGPT Prediction Engine
==========================

Dual-Prediction System for Dating Success Analytics

This script provides a user-friendly interface for predicting:
1. Meeting Probability: Likelihood of two people meeting based on demographic and platform data
2. Relationship Success: Probability of successful relationship for users who meet

Features:
- Loads pre-trained machine learning models
- Processes user profile data for compatibility analysis
- Generates dual probability predictions
- Provides actionable insights for dating decisions

Usage:
1. Fill in the user profile section with match characteristics
2. Optionally provide compatibility metrics for enhanced accuracy
3. Run script to receive probability predictions

Author: DenisBeliaev
Date: October 2025
Version: 1.0
"""

import pandas as pd
import joblib

# =============================================================================
# USER PROFILE INPUT SECTION
# =============================================================================
# Fill in the characteristics of your potential match below
# Replace the example values with your actual data for accurate predictions

# Basic Demographics
age = 27                          # Integer: 18-99 (e.g., 25, 30, 45)
height = '180-190'                # String: Height range in cm (e.g., '170-180', '160-170')
body_type = 'athletic'                # String: Physical build (slim, athletic, muscular, average, etc.)
region_origin = 'MiddleEast'         # String: Geographic origin (MiddleEast, EasternEurope, NorthAmerica, Asia, etc.)

# Professional & Platform Information
sphere_employment = 'it / tech'     # String: Employment sector (it / tech, healthcare / medicine, finance / banking / consulting, sales / marketing, etc.)
app_source = 'tinder'             # String: Dating app used (tinder, grindr, bumble, hinge, etc.)
chat_channel = 'instagram dm'     # String: Communication platform (whatsapp, snapchat, instagram dm, telegram, etc.)
activity = 'active'               # String: Activity level (active, not active)
# =============================================================================
# RELATIONSHIP COMPATIBILITY METRICS (Optional - for success prediction)
# =============================================================================
# These metrics are typically rated after initial interaction on a 0-5 scale:
# 0 = No sex (applicable for sex_quality only), 1 = Poor, 2 = Below average, 3 = Average, 4 = Good, 5 = Excellent

sex_quality = 3                   # Float: Sexual compatibility rating (0-5)
emotional_comfort = 3             # Float: Emotional connection comfort (1-5)
chemistry = 3                     # Float: Overall chemistry and spark (1-5)
communication_style = 3           # Float: Communication style match (1-5)
# =============================================================================

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


# --- Prediction Function ---
def predict_meet_success(age, height, body_type, region_origin,
                         sphere_employment, app_source, chat_channel, activity,
                         sex_quality=0, emotional_comfort=0, chemistry=0, communication_style=0):
    """
    Predict meeting probability and relationship success probability for a user profile.

    This function takes user demographic, platform usage, and compatibility metrics
    to generate dual predictions using pre-trained machine learning models.

    Args:
        age (int): User's age
        height (str): Height range in format 'low-high' (e.g., '170-180')
        body_type (str): Physical build description
        region_origin (str): Geographic origin region
        sphere_employment (str): Employment sector/industry
        app_source (str): Dating app used
        chat_channel (str): Primary communication channel
        activity (str): Fact of activity ("active"/"not active")
        sex_quality (float, optional): Sexual compatibility rating (0-5 scale)
        emotional_comfort (float, optional): Emotional comfort rating (0-5 scale)
        chemistry (float, optional): Chemistry rating (0-5 scale)
        communication_style (float, optional): Communication match rating (0-5 scale)

    Returns:
        tuple: (meeting_probability, success_probability) as floats between 0-1
    """

    # Convert height range string to numerical value (average of range)
    low, high = map(int, height.split('-'))
    height_num_value = (low + high) / 2

    # Calculate interaction feature used in training
    age_height_interaction = age * height_num_value

    # Prepare feature data for meeting prediction model
    meet_data = pd.DataFrame({
        'age': [age],
        'height_num': [height_num_value],
        'age_height_interaction': [age_height_interaction],  # ADD THIS LINE
        'body_type': [body_type],
        'region_origin': [region_origin],
        'sphere_of_employment': [sphere_employment],
        'app_source': [app_source],
        'chat_channel': [chat_channel],
        'activity': [activity]
    })

    # Generate meeting probability prediction (class 1 probability)
    meet_prob = meet_pipeline.predict_proba(meet_data)[0][1]

    # Initialize success probability
    success_prob = 0

    # Generate relationship success prediction if model is available
    if success_pipeline:
        success_data = pd.DataFrame({
            'age': [age],
            'height_num': [height_num_value],
            'age_height_interaction': [age_height_interaction],  # ADD THIS LINE
            'body_type': [body_type],
            'region_origin': [region_origin],
            'sphere_of_employment': [sphere_employment],
            'app_source': [app_source],
            'chat_channel': [chat_channel],
            'activity': [activity],
            'sex_quality': [sex_quality],
            'emotional_comfort': [emotional_comfort],
            'chemistry': [chemistry],
            'communication_style_match': [communication_style]
        })
        success_prob = success_pipeline.predict_proba(success_data)[0][1]

    return meet_prob, success_prob

# Generate predictions using the trained models
meet_prob, success_prob = predict_meet_success(
    age, height, body_type, region_origin, sphere_employment,
    app_source, chat_channel, activity,
    sex_quality=sex_quality, emotional_comfort=emotional_comfort,
    chemistry=chemistry, communication_style=communication_style
)

# Display prediction results with formatted output
print("\n📊 PREDICTION RESULTS:")
print(f"Meeting probability: {meet_prob * 100:.1f}%")
print(f"Relationship success probability: {success_prob * 100:.1f}%")