import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv('./MUSIC_GENRE_CLAS/music_genre.csv')

# Drop rows with missing values (where the whole row is missing)
df = df.dropna()

# Replace 'empty_field' in 'artist_name' with 'Unknown Artist'
df['artist_name'] = df['artist_name'].replace('empty_field', 'Unknown Artist')

# Replace negative duration values with the median of valid durations
median_duration = df[df['duration_ms'] > 0]['duration_ms'].median()
df['duration_ms'] = df['duration_ms'].replace(-1, median_duration)

# Replace missing tempo values
df['tempo'] = pd.to_numeric(df['tempo'].replace('?', np.nan), errors='coerce')
mean_tempo = df['tempo'].mean()
df['tempo'] = df['tempo'].fillna(mean_tempo)

# Encode categorical features
encoder = LabelEncoder()
df['key_encoded'] = encoder.fit_transform(df['key'])
df['mode_encoded'] = encoder.fit_transform(df['mode'])
df['music_genre_encoded'] = encoder.fit_transform(df['music_genre'])

# Drop original categorical columns and irrelevant columns
df = df.drop(columns=['key', 'mode', 'music_genre', 'instance_id', 'artist_name', 'track_name', 'obtained_date'])

# Drop mode_encoded and key_encoded
df = df.drop(columns=['mode_encoded', 'key_encoded'])

# Define features and target
y = df['music_genre_encoded']
X = df.drop(columns=['music_genre_encoded'])

# Function to train and evaluate Decision Tree with hyperparameter tuning
def train_decision_tree(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 10, 20, 30, 40, 50],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    
    print(f'Best parameters for Decision Tree: {grid_search.best_params_}')
    print(f'Best cross-validation accuracy for Decision Tree: {grid_search.best_score_}')
    
    best_tree_model = grid_search.best_estimator_
    y_pred_tree = best_tree_model.predict(X_test)
    
    print(classification_report(y_test, y_pred_tree))
    print(confusion_matrix(y_test, y_pred_tree))

# Function to train and evaluate Random Forest with hyperparameter tuning
def train_random_forest(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    
    print(f'Best parameters for Random Forest: {grid_search.best_params_}')
    print(f'Best cross-validation accuracy for Random Forest: {grid_search.best_score_}')
    
    best_forest_model = grid_search.best_estimator_
    y_pred_forest = best_forest_model.predict(X_test)
    
    print(classification_report(y_test, y_pred_forest))
    print(confusion_matrix(y_test, y_pred_forest))

# Function to train and evaluate XGBoost with hyperparameter tuning
def train_xgboost(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 4, 5, 6, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.7, 0.8, 0.9, 1.0]
    }
    
    grid_search = GridSearchCV(XGBClassifier(random_state=42), param_grid, cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    
    print(f'Best parameters for XGBoost: {grid_search.best_params_}')
    print(f'Best cross-validation accuracy for XGBoost: {grid_search.best_score_}')
    
    best_xgb_model = grid_search.best_estimator_
    y_pred_xgb = best_xgb_model.predict(X_test)
    
    print(classification_report(y_test, y_pred_xgb))
    print(confusion_matrix(y_test, y_pred_xgb))

# Train and evaluate models
#train_decision_tree(X, y)
#train_random_forest(X, y)
#train_xgboost(X, y)

# Define the individual models with the best parameters
decision_tree = DecisionTreeClassifier(
    criterion='entropy', max_depth=10, min_samples_leaf=1, min_samples_split=10, random_state=42)
random_forest = RandomForestClassifier(
    max_depth=10, min_samples_leaf=1, min_samples_split=5, n_estimators=300, random_state=42)
xgboost = XGBClassifier(
    learning_rate=0.2, max_depth=3, n_estimators=200, subsample=0.7, random_state=42)

# Combine the models using a Voting Classifier
voting_model = VotingClassifier(
    estimators=[('dt', decision_tree), ('rf', random_forest), ('xgb', xgboost)],
    voting='soft')  # Use 'hard' for majority voting, 'soft' for weighted voting

# Split the data again for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Voting Classifier
voting_model.fit(X_train, y_train)

# Predict and evaluate
y_pred_voting = voting_model.predict(X_test)
print(classification_report(y_test, y_pred_voting))
print(confusion_matrix(y_test, y_pred_voting))
test_acc_voting = accuracy_score(y_test, y_pred_voting)
print(f'Test accuracy of Voting Classifier: {test_acc_voting}')
