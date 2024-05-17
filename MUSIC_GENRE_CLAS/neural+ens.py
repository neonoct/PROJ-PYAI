import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from scikeras.wrappers import KerasClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score


# Load the dataset
df = pd.read_csv('./MUSIC_GENRE_CLAS/music_genre.csv')

# Drop rows with missing values (where the whole row is missing)
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
    #print which genre corresponds to which encoded value
    print(df[['music_genre', 'music_genre_encoded']].drop_duplicates().sort_values(by='music_genre_encoded'))
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

# Define the best neural network model function
def create_best_model(optimizer='rmsprop', neurons=128, dropout_rate=0.3):
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

# Custom wrapper for KerasClassifier to handle integer labels
class MyKerasClassifier(KerasClassifier):
    def fit(self, X, y, **kwargs):
        return super().fit(X, to_categorical(y), **kwargs)
    
    def predict(self, X, **kwargs):
        pred = super().predict(X, **kwargs)
        return np.argmax(pred, axis=1)

# Create the KerasClassifier for the best neural network
best_nn_model = MyKerasClassifier(model=create_best_model, optimizer='rmsprop', neurons=128, dropout_rate=0.3, epochs=100, batch_size=64, verbose=2)

# Define the RandomForest and XGBoost models
rf_model = RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_split=5, min_samples_leaf=1, random_state=42)
xgb_model = XGBClassifier(learning_rate=0.2, max_depth=3, n_estimators=200, subsample=0.7, random_state=42)
svm_model = SVC(C=10, gamma='scale', kernel='rbf', probability=True, random_state=42)

#only the neural network

#Cross-validation accuracy: 59.02%
#Test accuracy of Neural Network: 60.07%
#difference between cross-validation and test accuracy is 0.0105

# Create the VotingClassifier with neural network + XGBoost
#voting_model = VotingClassifier(estimators=[('nn', best_nn_model),  ('xgb', xgb_model)], voting='soft')
#Cross-validation accuracy: 0.5934 ± 0.0039
#Test accuracy of Voting Classifier: 0.5987 
#difference between cross-validation and test accuracy is 0.0053

#Create the VotingClassifier with Neural Network + SVM
voting_model = VotingClassifier(estimators=[('nn', best_nn_model), ('svm', svm_model)], voting='soft')
#Cross-validation accuracy: 0.5912 ± 0.0029
#Test accuracy of Voting Classifier: 0.594
#difference between cross-validation and test accuracy is 0.0028

# Create the VotingClassifier with neural network + XGBoost + SVM
#voting_model = VotingClassifier(estimators=[('nn', best_nn_model), ('xgb', xgb_model), ('svm', svm_model)], voting='soft')
#Cross-validation accuracy:59.53% ± 0.31%
#Test accuracy of Voting Classifier: 0.5957
#difference between cross-validation and test accuracy is 0.0044





# Cross-validation
cv_scores = cross_val_score(voting_model, X_train, y_train, cv=5, scoring='accuracy')
print(f'Cross-validation accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}')

# Fit the VotingClassifier (ensure y_train is not one-hot encoded)
voting_model.fit(X_train, y_train)

# Predict and evaluate
y_pred_voting = voting_model.predict(X_test)
print(f'Test accuracy of Voting Classifier: {accuracy_score(y_test, y_pred_voting)}')
print(classification_report(y_test, y_pred_voting))
print(confusion_matrix(y_test, y_pred_voting))


