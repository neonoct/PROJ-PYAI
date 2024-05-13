import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import chi2_contingency
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv('./MUSIC_GENRE_CLAS/music_genre.csv')

# Drop rows with missing values(where the whole row is missing)
df = df.dropna()

def replace_empty_artist_name(df):
    df['artist_name'] = df['artist_name'].replace('empty_field', 'Unknown Artist')
    return df

replace_empty_artist_name(df)

def replace_negative_duration(df):
    # Calculate the median duration from valid entries
    median_duration = df[df['duration_ms'] > 0]['duration_ms'].median()

    # Replace -1 values with the median duration
    df['duration_ms'] = df['duration_ms'].replace(-1, median_duration)
    # Optional: Cap durations at the 95th percentile to limit the impact of extreme outliers
    #percentile_95 = df['duration_ms'].quantile(0.95)
    #df['duration_ms'] = df['duration_ms'].clip(upper=percentile_95)
    
    return df

df = replace_negative_duration(df)

def replace_missing_tempo(df):
    df['tempo'] = pd.to_numeric(df['tempo'].replace('?', np.nan), errors='coerce')
    mean_tempo = df['tempo'].mean()
    df['tempo'] = df['tempo'].fillna(mean_tempo)
    return df

replace_missing_tempo(df)

def fill_missing_tempo(df):
    #this aproach allows for more nuanced imputation of missing tempo values based on the music genre respects the typical characteristics of each genre
    if 'music_genre' in df.columns:
        for genre in df['music_genre'].unique():
            genre_mode = df[df['music_genre'] == genre]['tempo'].mode()[0]
            df.loc[(df['tempo'].isnull()) & (df['music_genre'] == genre), 'tempo'] = genre_mode

fill_missing_tempo(df)

def encode_categorical_features(df):
    # Assuming 'key' and 'mode' are categorical features in your dataset
    encoder = LabelEncoder()
    df['key_encoded'] = encoder.fit_transform(df['key'])
    df['mode_encoded'] = encoder.fit_transform(df['mode'])
    df['music_genre_encoded'] = encoder.fit_transform(df['music_genre'])  # ensure it's encoded for this usage
    return df

encode_categorical_features(df)

#drop the original categorical columns
df = df.drop(columns=['key', 'mode', 'music_genre'])

#drop insance_id,artist_name,track_name,obtained_date
df = df.drop(columns=['instance_id', 'artist_name', 'track_name', 'obtained_date'])

#drop mode_encoded and key_encoded
df = df.drop(columns=['mode_encoded', 'key_encoded'])

y=df['music_genre_encoded']
X=df.drop(columns=['music_genre_encoded'])

def train_decision_tree(X, y):
    # Assuming 'X' as features and 'y' as labels
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the Decision Tree
    tree_model = DecisionTreeClassifier(random_state=42)
    tree_model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred_tree = tree_model.predict(X_test)
    print(classification_report(y_test, y_pred_tree))
    print(confusion_matrix(y_test, y_pred_tree))

# Call the function
train_decision_tree(X, y)

