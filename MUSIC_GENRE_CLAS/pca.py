import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import LabelEncoder
from scipy.stats import chi2_contingency
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

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

#Feature Engineering################################
####################################################

#Creating new features from existing ones##

#interaction features

#show head of the dataset with only the las 15 columns



def create_interaction_features(df):
    df['energy_danceability'] = df['energy'] * df['danceability']
    
    scaler = StandardScaler()#standar scaler scales tjese features to have a mean of 0 and a standard deviation of 1
    df['loudness_scaled'] = scaler.fit_transform(df[['loudness']])
    df['loudness_energy'] = df['loudness_scaled'] * df['energy']
    
    return df

#df = create_interaction_features(df)



#aggregate features

def create_acoustic_instrumental_ratio(df):
    df['acoustic_instrumental_ratio'] = df['acousticness'] / (df['instrumentalness'] + 0.001)
    return df

#create_acoustic_instrumental_ratio(df)

#Categorical Binning of Continuous Variables

def create_categorical_features(df):
    bins = [0, 60, 90, 120, 150, 180, float('inf')]
    labels = ['very_slow', 'slow', 'moderate', 'fast', 'very_fast', 'extremely_fast']
    df['tempo_category'] = pd.cut(df['tempo'], bins=bins, labels=labels)

    df['duration_cat'] = pd.cut(df['duration_ms'], bins=[0, 180000, 240000, float('inf')], labels=['short', 'medium', 'long'])
    
    return df

#create_categorical_features(df)



def generate_polynomial_features(df):

    # Initialize the PolynomialFeatures object with degree 2 (for quadratic interactions)
    poly = PolynomialFeatures(degree=2, include_bias=False)

    # Select features to transform
    features = df[['tempo', 'energy', 'danceability', 'loudness', 'acousticness']]

    # Scale features before applying polynomial transformations
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Generate polynomial features
    features_poly = poly.fit_transform(features_scaled)
    poly_feature_names = poly.get_feature_names_out(['tempo', 'energy', 'danceability', 'loudness', 'acousticness'])

    # Create a DataFrame with the new polynomial features
    df_poly = pd.DataFrame(features_poly, columns=poly_feature_names)

    # Reset indices if they do not match
    df.reset_index(drop=True, inplace=True)
    df_poly.reset_index(drop=True, inplace=True)

    # Drop original feature columns from df_poly
    original_features = ['tempo', 'energy', 'danceability', 'loudness', 'acousticness']
    df_poly.drop(original_features, axis=1, inplace=True)

    # Merge the new polynomial features back into the original DataFrame
    df = pd.concat([df, df_poly], axis=1)
    
    return df
    
    

#display the features before adding polynomial features (whether they are categorical or continuous)
#print(df.dtypes)

#db info


#df = generate_polynomial_features(df)#total number of polynomial features generated is 21



#or if it is the second time with the same column name
new_column_names = []
for col in df.columns:
    #or if it is the second time with the same column name
    if '^' in col or ' ' in col:  # Identify polynomial feature columns
        

        new_column_names.append(col + '_poly')  # Append a suffix to denote polynomial features
    else:
        new_column_names.append(col)

df.columns = new_column_names

#added polynomial features to the dataset i do not know if  they are useful or not but i will keep them for now

#########################################
#EDA- exploratory data analysis 

#Chi-Squared Test for Feature Selection -catgorical features
def encode_categorical_features(df):
    # Assuming 'key' and 'mode' are categorical features in your dataset
    encoder = LabelEncoder()
    df['key_encoded'] = encoder.fit_transform(df['key'])
    df['mode_encoded'] = encoder.fit_transform(df['mode'])
    df['music_genre_encoded'] = encoder.fit_transform(df['music_genre'])  # ensure it's encoded for this usage
    return df

encode_categorical_features(df)

#show head of the dataset with only the las 15 columns


#describe the dataset




#model-based feature selection-random forest



#model-based feature selection-random forest
def feature_ranking(df):
    # Dropping non-useful non-numeric columns
    X = df.drop(['music_genre_encoded', 'artist_name', 'track_name', 'obtained_date', 'key', 'mode', 'music_genre','instance_id'], axis=1)

    # Assuming 'tempo_category' and 'duration_cat' need to be encoded if they haven't been already
    if 'tempo_category' in X.columns:
        X['tempo_category_encoded'] = LabelEncoder().fit_transform(X['tempo_category'])
        X.drop('tempo_category', axis=1, inplace=True)
    if 'duration_cat' in X.columns:
        X['duration_cat_encoded'] = LabelEncoder().fit_transform(X['duration_cat'])
        X.drop('duration_cat', axis=1, inplace=True)

    y = df['music_genre_encoded']

    # Splitting the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and fit the random forest
    forest = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        max_features='sqrt',
        min_samples_leaf=1,
        min_samples_split=5,
        random_state=42
    )
    forest.fit(X_train, y_train)

    # Predict on the test set
    y_pred = forest.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")

    # Get feature importances
    importances = forest.feature_importances_
    indices = np.argsort(importances)[::-1]

    #Print the feature rankings
    print("Feature ranking:")
    for f in range(X.shape[1]):
        print(f"{f + 1}. feature {X.columns[indices[f]]} ({importances[indices[f]]})")
    
    return accuracy # Return the accuracy for comparison

feature_ranking(df)
#dropped columns are: 'music_genre_encoded', 'artist_name', 'track_name', 'obtained_date', 'key', 'mode', 'music_genre'

#describe the dataset
print('after feature ranking');



def train_random_forest(df):
    important_features = [
        'popularity','loudness','instrumentalness','speechiness',
        'acousticness','danceability','energy','valence',
        'duration_ms','tempo'
    ]
    #tried with other combinations too but this one gave the best accuracy

    

    X = df[important_features]
    y = df['music_genre_encoded']

    # Splitting the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the random forest on the refined feature set
    #initialize the random forest classifier with the hyperparameters that were found to be optimal
    forest_refined = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        max_features='sqrt',
        min_samples_leaf=1,
        min_samples_split=5,
        random_state=42
    )
    forest_refined.fit(X_train, y_train)

    # Predict on the test set
    y_pred = forest_refined.predict(X_test)

    # Calculate accuracy
    accuracy_refined = accuracy_score(y_test, y_pred)
    print(f"Accuracy with refined features: {accuracy_refined:.2f}")

    # Compare this accuracy with the previous model's accuracy
    return accuracy_refined

train_random_forest(df)
#improvement was not significant from 0.53 to 0.54

def train_random_forest_with_hyperparameter_tuning(df):
    # Assuming the important features and encoded target are already defined
    X = df[[
        'popularity','loudness','instrumentalness','speechiness',
        'acousticness','danceability','energy','valence',
        'duration_ms','tempo'
            ]]
    y = df['music_genre_encoded']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Set the parameters by cross-validation
    param_grid = {
        'n_estimators': [100, 200, 300],  # Number of trees in random forest
        'max_features': ['sqrt'],  # Number of features to consider at every split
        'max_depth': [10, 20, 30, None],  # Maximum number of levels in tree
        'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split a node
        'min_samples_leaf': [1, 2, 4]  # Minimum number of samples required at each leaf node
    }

    # Initialize the classifier
    rf = RandomForestClassifier(random_state=42)

    # Initialize the GridSearchCV object
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, verbose=2, n_jobs=-1)

    # Fit the grid search to the data
    grid_search.fit(X_train, y_train)

    # Best parameters found
    print("Best parameters found: ", grid_search.best_params_)

    # Evaluate the best grid search model on the test set
    best_grid = grid_search.best_estimator_
    y_pred = best_grid.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy with the best parameters: {accuracy:.2f}")

#train_random_forest_with_hyperparameter_tuning(df)
#Best parameters found:  {'max_depth': 10, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 300}
#Accuracy: 0.53
#Accuracy with refined features: 0.56 - after hyperparameter tuning for the random forest classifier




