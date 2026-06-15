import re
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def enhance_features(df):
    # make a copy so original dataframe is not modified
    df = df.copy()
    # handle missing comments
    df['comment'] = df['comment'].fillna("none")
    df['comment'] = df['comment'].astype(str)

    # Fill categorical missing values
    categorical_cols = ["race", "religion", "gender"]
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].fillna("unknown")

    # Fill boolean missing values
    if "disability" in df.columns:
        df["disability"] = df["disability"].fillna(False)

    char_counts = []
    caps_ratios = []
    exclamation_counts = []
    cleaned_comments = []
    for text in df['comment']:
        # character count
        char_counts.append(len(text))
        # uppercase ratio
        upper_chars = sum(1 for c in text if c.isupper())
        caps_ratios.append(upper_chars / (len(text) + 1))
        # number of exclamation marks
        exclamation_counts.append(text.count('!'))
        # remove URLs
        cleaned = re.sub(r'http\S+|www\S+|https\S+', '', text)
        # remove non alphabet characters
        cleaned = re.sub(r'[^a-zA-Z\s]', ' ', cleaned)
        # convert to lowercase
        cleaned = cleaned.lower()
        cleaned_comments.append(cleaned)
    # assign new columns
    df['char_count'] = char_counts
    df['caps_ratio'] = caps_ratios
    df['excl_count'] = exclamation_counts
    df['clean_text'] = cleaned_comments
    return df

# Columns for the transformer
num_cols = ['upvote', 'downvote', 'if_1', 'if_2', 'char_count', 'caps_ratio', 'excl_count']
cat_cols = ['race', 'religion', 'gender', 'disability']

preprocessor = ColumnTransformer([
    ('text', TfidfVectorizer(max_features=30000, ngram_range=(1, 3), sublinear_tf=True), 'clean_text'),
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
])
