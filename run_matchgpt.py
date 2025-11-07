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

    # Prepare feature data for meeting prediction model
    meet_data = pd.DataFrame({
        'age': [age],
        'height_num': [height_num_value],
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