import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

# Replace 'empty_field' with 'Unknown Artist'
df['artist_name'] = df['artist_name'].replace('empty_field', 'Unknown Artist') # for this project i will assume that artist_name is not important and i will not use(probably) it in the model

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
    
#we are going to use the median to replace the -1 values in the duration_ms column instead of the mean because the data is right-skewed
# Calculate the median duration from valid entries
median_duration = df[df['duration_ms'] > 0]['duration_ms'].median()

# Replace -1 values with the median duration
df['duration_ms'] = df['duration_ms'].replace(-1, median_duration)
# Optional: Cap durations at the 95th percentile to limit the impact of extreme outliers
#percentile_95 = df['duration_ms'].quantile(0.95)
#df['duration_ms'] = df['duration_ms'].clip(upper=percentile_95)

def analyze_instrumentalness(df):#understood that 0s are valid values in the instrumentalness column
    # Check the distribution of other features where instrumentalness is 0
    print(df[df['instrumentalness'] == 0].describe())

    # Display summary statistics for numeric columns-for comparison
    print(df.describe())


# Replace '?' with NaN and convert the column to float in the tempo column
df['tempo'] = pd.to_numeric(df['tempo'].replace('?', np.nan), errors='coerce')#after converting the tempo column to float, we can now calculate the mean tempo and replace the missing values with it

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

df['energy_danceability'] = df['energy'] * df['danceability']#no need to scale as the features are already on the same scale 0-1 -this category mostly useful for detecting genres like electronic

scaler = StandardScaler()
# Assuming 'loudness' needs to be scaled for interaction with 'energy'
df['loudness_scaled'] = scaler.fit_transform(df[['loudness']])
df['loudness_energy'] = df['loudness_scaled'] * df['energy']

#aggregate features

df['acoustic_instrumental_ratio'] = df['acousticness'] / (df['instrumentalness'] + 0.001)#adding a small value to the denominator to avoid division by zero

#Categorical Binning of Continuous Variables
bins = [0, 60, 90, 120, 150, 180, float('inf')]
labels = ['very_slow', 'slow', 'moderate', 'fast', 'very_fast', 'extremely_fast']
df['tempo_category'] = pd.cut(df['tempo'], bins=bins, labels=labels)

df['duration_cat'] = pd.cut(df['duration_ms'], bins=[0, 180000, 240000, float('inf')], labels=['short', 'medium', 'long'])

#polynomial features

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
# Check how many rows have all values as NaN
missing_data_count = df.isnull().all(axis=1).sum()
print(f"Total rows completely missing: {missing_data_count}")

# Detailed missing data count for each column
print(df.isnull().sum())

# Reset indices if they do not match
df.reset_index(drop=True, inplace=True)
df_poly.reset_index(drop=True, inplace=True)


# Merge the new polynomial features back into the original DataFrame
df = pd.concat([df, df_poly], axis=1)























