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

data = pd.read_csv("match_gpt_testset.csv")
