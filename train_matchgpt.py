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

# Title,age,height,body_type,region_origin,sphere_of_employment,app_source,chat_channel,met,days_to_meet,sex_quality,emotional_comfort,chemistry,communication_style_match,relationship_outcome,activity
data = pd.read_csv("match_gpt_testset.csv")

def process_height(height):
    if isinstance(height, str) and '-' in height:
        low, high = map(int, height.split('-'))
        return (low + high) / 2
    return np.nan

data['height_num'] = data['height'].apply(process_height)
data['height_num'] = data['height_num'].fillna(data['height_num'].median())
