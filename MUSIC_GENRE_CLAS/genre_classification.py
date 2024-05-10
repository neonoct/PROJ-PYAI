import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures


# Load the dataset
df = pd.read_csv('./MUSIC_GENRE_CLAS/music_genre.csv')


def explore_dataset(df):
    # Display the first few rows of the DataFrame
    #print(df.head())

    # Display info about data types and number of non-null values
    print(df.info()) # some of the tempo values seem to be strings it is because missing values in this column are represented as '?' in the dataset

    # Display summary statistics for numeric columns
    print(df.describe())

    # Optionally, explore the number of unique values in categorical columns
    for col in df.select_dtypes(include=['object']).columns: 
        print(f"{col} has {df[col].nunique()} unique values")

    # Check for missing values
    print(df.isnull().sum()) # 5 rows are missing


# Drop rows with missing values(where the whole row is missing)
df = df.dropna()

def replace_empty_artist_name(df):
    df['artist_name'] = df['artist_name'].replace('empty_field', 'Unknown Artist')
    return df

replace_empty_artist_name(df)

def analyze_duration(df):
    # Decide whether to use median or mean to replace -1 values in duration_ms
    # First, you should examine the distribution of the duration_ms data. You can visualize the distribution using a histogram or a box plot and also calculate the skewness of the data.
    # Let's plot the histogram of the duration_ms column excluding -1 values
    df[df['duration_ms'] > 0]['duration_ms'].hist(bins=1000)
    plt.title('Distribution of duration_ms')
    plt.xlabel('Duration (ms)')
    plt.ylabel('Frequency')
    plt.show()

    # Calculate skewness
    skewness = df[df['duration_ms'] > 0]['duration_ms'].skew()
    print(f"Skewness of duration_ms: {skewness}")
    
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

def analyze_instrumentalness(df):#understood that 0s are valid values in the instrumentalness column
    # Check the distribution of other features where instrumentalness is 0
    print(df[df['instrumentalness'] == 0].describe())

    # Display summary statistics for numeric columns-for comparison
    print(df.describe())


def replace_missing_tempo(df):
    df['tempo'] = pd.to_numeric(df['tempo'].replace('?', np.nan), errors='coerce')
    mean_tempo = df['tempo'].mean()
    df['tempo'] = df['tempo'].fillna(mean_tempo)
    return df

replace_missing_tempo(df)

def analyze_tempo(df):
    df[df['tempo'] > 0]['tempo'].hist(bins=100)
    plt.title('Distribution of tempo')
    plt.xlabel('Tempo')
    plt.ylabel('Frequency')
    plt.show()

    # Calculate skewness
    skewness = df[df['tempo'] > 0]['tempo'].skew()
    print(f"Skewness of tempo: {skewness}")



def fill_missing_tempo(df):
    #this aproach allows for more nuanced imputation of missing tempo values based on the music genre respects the typical characteristics of each genre
    if 'music_genre' in df.columns:
        for genre in df['music_genre'].unique():
            genre_mode = df[df['music_genre'] == genre]['tempo'].mode()[0]
            df.loc[(df['tempo'].isnull()) & (df['music_genre'] == genre), 'tempo'] = genre_mode

fill_missing_tempo(df)

#########################################
#Creating new features from existing ones

#interaction features

def create_interaction_features(df):
    df['energy_danceability'] = df['energy'] * df['danceability']
    
    scaler = StandardScaler()
    df['loudness_scaled'] = scaler.fit_transform(df[['loudness']])
    df['loudness_energy'] = df['loudness_scaled'] * df['energy']
    
    return df

df = create_interaction_features(df)

#aggregate features

def create_acoustic_instrumental_ratio(df):
    df['acoustic_instrumental_ratio'] = df['acousticness'] / (df['instrumentalness'] + 0.001)
    return df

create_acoustic_instrumental_ratio(df)

#Categorical Binning of Continuous Variables

def create_categorical_features(df):
    bins = [0, 60, 90, 120, 150, 180, float('inf')]
    labels = ['very_slow', 'slow', 'moderate', 'fast', 'very_fast', 'extremely_fast']
    df['tempo_category'] = pd.cut(df['tempo'], bins=bins, labels=labels)

    df['duration_cat'] = pd.cut(df['duration_ms'], bins=[0, 180000, 240000, float('inf')], labels=['short', 'medium', 'long'])
    
    return df

create_categorical_features(df)

#polynomial features

def generate_polynomial_features(df):
    # Initialize the PolynomialFeatures object with degree 2 (for quadratic interactions)
    poly = PolynomialFeatures(degree=2, include_bias=False)

    # Select features to transform
    features = df[['tempo', 'energy', 'danceability', 'loudness', 'acousticness']]

    # It's a good practice to scale features before applying polynomial transformations
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

    # Merge the new polynomial features back into the original DataFrame
    df = pd.concat([df, df_poly], axis=1)
    
    return df

df = generate_polynomial_features(df)

#added polynomial features to the dataset i do not know if  they are useful or not but i will keep them for now

#########################################
#EDA- exploratory data analysis 

def plot_feature_distribution(df, feature):
    # Plot the distribution of a feature
    df[feature].hist(bins=100)
    plt.title(f'Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.show()

def plot_feature_boxplot(df, feature):
    # Plot a boxplot of a feature
    sns.boxplot(data=df, x=feature)
    plt.title(f'Boxplot of {feature}')
    plt.xlabel(feature)
    plt.show()

def plot_feature_by_genre(df, feature):
    # Plot the distribution of a feature by genre(target variable)
    plt.figure(figsize=(12, 6))
    sns.violinplot(data=df, x='music_genre', y=feature)
    plt.title(f'{feature} by Genre')
    plt.xlabel('Genre')
    plt.ylabel(feature)
    plt.xticks(rotation=45)
    plt.show()

# Histograms for continuous features
def plot_histograms(df):
    for col in df.select_dtypes(include=['float64']).columns:
        plot_feature_distribution(df, col)

# Boxplots for visualizing outliers
def plot_boxplots(df):
    for col in df.select_dtypes(include=['float64']).columns:
        plt.figure(figsize=(10, 4))
        sns.boxplot(x=df[col])
        plt.title(f'Boxplot of {col}')
        plt.xlabel(col)
        plt.show()

# Histograms for continuous features--via cptt
def plot_histograms(df):
    for col in df.select_dtypes(include=['float64']).columns:   
        plt.figure(figsize=(10, 4))
        sns.histplot(df[col], kde=True)
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.show()

# Boxplots for visualizing outliers--via cptt
def plot_boxplots(df):
    for col in df.select_dtypes(include=['float64']).columns:
        plt.figure(figsize=(10, 4))
        sns.boxplot(x=df[col])
        plt.title(f'Boxplot of {col}')
        plt.xlabel(col)
        plt.show()  

# Scatter plots for continuous features vs. music genre--via cptt
def plot_scatterplots(df):
    for col in df.select_dtypes(include=['float64']).columns:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=df[col], y=df['music_genre'])
        plt.title(f'Scatter plot of {col} vs. Music Genre')
        plt.xlabel(col)
        plt.ylabel('Music Genre')
        plt.show()

# Pairplot for selected features--via cptt
def plot_pairplot(df):
    features = ['tempo', 'energy', 'danceability', 'loudness', 'acousticness', 'music_genre']
    sns.pairplot(df[features], hue='music_genre', diag_kind='hist')
    #sns.pairplot(df[features], hue='music_genre', corner=True) # corner=True from cptt
    plt.show()


def visualize_correlation(df):
    # Correlation matrix
    correlation_matrix = df.corr()

    # Heatmap to visualize the correlation matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('Correlation Matrix of Features')
    plt.show()

    # Specific correlation with target, if target is numerically encoded; otherwise, consider ANOVA or similar tests
    # Assuming 'music_genre' is numerically encoded for this purpose
    if 'music_genre' in correlation_matrix:
        genre_corr = correlation_matrix['music_genre'].sort_values(ascending=False)
        print(genre_corr)





