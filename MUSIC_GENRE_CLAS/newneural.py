import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from scikeras.wrappers import KerasClassifier

# Load the dataset
df = pd.read_csv('./MUSIC_GENRE_CLAS/music_genre.csv')

# Drop rows with missing values (where the whole row is missing)
df is None
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
    return df

df = replace_negative_duration(df)

def replace_missing_tempo(df):
    df['tempo'] = pd.to_numeric(df['tempo'].replace('?', np.nan), errors='coerce')
    mean_tempo = df['tempo'].mean()
    df['tempo'] = df['tempo'].fillna(mean_tempo)
    return df

replace_missing_tempo(df)

def fill_missing_tempo(df):
    if 'music_genre' in df.columns:
        for genre in df['music_genre'].unique():
            genre_mode = df[df['music_genre'] == genre]['tempo'].mode()[0]
            df.loc[(df['tempo'].isnull()) & (df['music_genre'] == genre), 'tempo'] = genre_mode

fill_missing_tempo(df)

def encode_categorical_features(df):
    encoder = LabelEncoder()
    df['key_encoded'] = encoder.fit_transform(df['key'])
    df['mode_encoded'] = encoder.fit_transform(df['mode'])
    df['music_genre_encoded'] = encoder.fit_transform(df['music_genre'])
    return df

encode_categorical_features(df)

# Drop the original categorical columns
df = df.drop(columns=['key', 'mode', 'music_genre'])

# Drop irrelevant columns
df = df.drop(columns=['instance_id', 'artist_name', 'track_name', 'obtained_date'])

# Drop encoded columns
df = df.drop(columns=['mode_encoded', 'key_encoded'])

y = df['music_genre_encoded']
X = df.drop(columns=['music_genre_encoded'])

# Normalize/standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Convert labels to categorical one-hot encoding
y_train_categorical = to_categorical(y_train)
y_test_categorical = to_categorical(y_test)

# Define a function to create the model (required for KerasClassifier)
def create_model(optimizer='adam', neurons=64, dropout_rate=0.5):
    model = Sequential([
        Dense(neurons, activation='relu', input_shape=(X_train.shape[1],)),
        BatchNormalization(),
        Dropout(dropout_rate),
        Dense(neurons, activation='relu'),
        BatchNormalization(),
        Dropout(dropout_rate),
        Dense(neurons, activation='relu'),
        BatchNormalization(),
        Dropout(dropout_rate),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Create the KerasClassifier
model = KerasClassifier(model=create_model, verbose=2)

# Define the grid of hyperparameters to search
param_grid = {
    'model__optimizer': ['adam', 'rmsprop'],
    'model__neurons': [32, 64, 128],
    'model__dropout_rate': [0.3, 0.5, 0.7],
    'batch_size': [32, 64],
    'epochs': [50, 100]
}

# Randomized Search
random_search = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=10, cv=3, verbose=2, n_jobs=-1)

# Fit the random search model
random_search_result = random_search.fit(X_train, y_train_categorical)

# Print the best parameters and accuracy
print(f'Best parameters: {random_search_result.best_params_}')
print(f'Best cross-validation accuracy: {random_search_result.best_score_}')

# Evaluate the best model on the test set
best_model = random_search_result.best_estimator_
y_pred_best_model = best_model.predict(X_test)

# Convert one-hot encoded predictions back to integer labels
y_pred_best_model = np.argmax(y_pred_best_model, axis=1)

# Print the evaluation metrics
print(f'Test accuracy of best model: {accuracy_score(y_test, y_pred_best_model)}')
print(classification_report(y_test, y_pred_best_model))
print(confusion_matrix(y_test, y_pred_best_model))
