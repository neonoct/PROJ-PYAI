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
from xgboost import XGBClassifier
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score


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

# Normalize/standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define the kNN model
knn_model = KNeighborsClassifier(n_neighbors=5)

# Fit the model
knn_model.fit(X_train, y_train)

# Predict on the test set
y_pred = knn_model.predict(X_test)

# Calculate accuracy
test_acc_knn = accuracy_score(y_test, y_pred)
print(f'Test accuracy of kNN: {test_acc_knn}')

# Try different values of k
k_values = range(1, 21)
accuracies = []

for k in k_values:
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(X_train, y_train)
    y_pred = knn_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)

# Plot the accuracy for different k values
plt.figure(figsize=(10, 6))
plt.plot(k_values, accuracies, marker='o')
plt.title('kNN Accuracy for Different k Values')
plt.xlabel('Number of Neighbors k')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()

# Different values of k and distance metrics to test
k_values = range(1, 21)
metrics = ['euclidean', 'manhattan', 'chebyshev']

# Dictionary to store the accuracy results
results = {}

for metric in metrics:
    accuracies = []
    for k in k_values:
        knn_model = KNeighborsClassifier(n_neighbors=k, metric=metric, weights='distance')
        knn_model.fit(X_train, y_train)
        y_pred = knn_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)
    results[metric] = accuracies

# Plot the accuracy for different k values and metrics
plt.figure(figsize=(10, 6))
for metric in metrics:
    plt.plot(k_values, results[metric], marker='o', label=f'Metric: {metric}')
plt.title('kNN Accuracy for Different k Values and Distance Metrics')
plt.xlabel('Number of Neighbors k')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# Apply PCA to reduce the dimensionality
pca = PCA(n_components=10)  # Adjust the number of components as needed
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Define the kNN model with PCA-transformed data
knn_model_pca = KNeighborsClassifier(n_neighbors=5)
knn_model_pca.fit(X_train_pca, y_train)

# Predict on the test set
y_pred_pca = knn_model_pca.predict(X_test_pca)

# Calculate accuracy
test_acc_knn_pca = accuracy_score(y_test, y_pred_pca)
print(f'Test accuracy of kNN with PCA: {test_acc_knn_pca}')

# Function to apply PCA and evaluate kNN
def evaluate_knn_with_pca(n_components):
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    
    knn_model_pca = KNeighborsClassifier(n_neighbors=5)
    knn_model_pca.fit(X_train_pca, y_train)
    
    y_pred_pca = knn_model_pca.predict(X_test_pca)
    return accuracy_score(y_test, y_pred_pca)

# Test different numbers of PCA components
components_range = range(5, X_train.shape[1] + 1)
accuracies_pca = [evaluate_knn_with_pca(n) for n in components_range]

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(components_range, accuracies_pca, marker='o')
plt.title('kNN Accuracy with Different Numbers of PCA Components')
plt.xlabel('Number of PCA Components')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()

# Define the kNN model with the best parameters found so far
best_knn_model = KNeighborsClassifier(n_neighbors=15, metric='manhattan', weights='distance')

# Perform cross-validation
cv_scores = cross_val_score(best_knn_model, X_scaled, y, cv=5)

# Calculate and print the average accuracy
average_cv_accuracy = cv_scores.mean()
print(f'Average cross-validated accuracy of kNN: {average_cv_accuracy}')

