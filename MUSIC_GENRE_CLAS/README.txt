BISMILLAHIRRAHMANIRRAHIM
__________________________
explore_dataset(df)

# Check how many rows have all values as NaN
missing_data_count = df.isnull().all(axis=1).sum()
print(f"Total rows completely missing: {missing_data_count}")

# Detailed missing data count for each column
print(df.isnull().sum())

Prediction of music genre
Classify music into genres

The full list of genres included in the CSV are 'Electronic', 'Anime', 'Jazz', 'Alternative', 'Country', 'Rap', 'Blues', 'Rock', 'Classical', 'Hip-Hop'.

https://www.kaggle.com/datasets/vicsuperman/prediction-of-music-genre

instance_id: Likely an identifier for each record in the dataset.
popularity: Numeric value representing the popularity of a track.
acousticness: Measure of how acoustic a track is.
danceability: Measure of how suitable a track is for dancing based on a combination of musical elements.
duration_ms: Duration of the track in milliseconds.
energy: Measure of intensity and activity.
instrumentalness: Indicates the likelihood that a track contains no vocals.
liveness: Detects the presence of an audience in the recording.
loudness: The overall loudness of a track in decibels (dB).
speechiness: Measures the presence of spoken words in a track.
valence: Measure of the musical positiveness conveyed by a track.

1-some songs are in two separate rows for being in two different genres(e.g-Blockhead-Cheer up, You're Not Dead Yet" is both in electronic and jazz)-total 7k
 (also their popularity are different in each instance as well as their genres)
Total songs with multiple genres: 3633

2-2,5k of the songs has empty_field as artist_name
3-5 rows are empty
4-all the song names are valid but some of them are in different Fonts 
5-some of the acousticness is in scientific notation(e.g 4.18e-05) but there is no problem for them they are directly read 
6-no problem seen with danceability
7-some duration_ms is -1 meaning empty cell--5k columns
8-some instrumentalness values are 0(missing data)-15K columns( are they missing or their instrumentalness is just 0 ?)
9-some tempo values are missing ? -5K they are also categorized under object(string) because a part of the data is with '?' normally it should be numerical data(float64)
10-from every label(genre) 5k instances
11-obtained date can be dropped it is probably about on which date the data about that instance is gathered into the dataset
12-valence means how likely is the song to be positive higher the value positiver the song-2 songs have 0 valence

--changing artist_name s that are empty_field to Unknown artist

---analyzing duration_ms{Distribution Characteristics:
Central Peak: Most of the track durations are clustered around a central peak, which appears to be around 200,000 to 300,000 milliseconds (200 to 300 seconds). This range likely represents typical song lengths.
Long Tail: There is a long tail extending to the right, up to about 800,000 milliseconds (800 seconds), indicating that there are some tracks significantly longer than the average.
No Negative Values: The updated histogram scale starts from 0 and shows no negative values, providing a more realistic view of the data.
Skewness:
Value of 7.48: The high skewness value (greater than 1) confirms the visual assessment of the distribution being right-skewed. It indicates that the mean of the distribution is greater than the median due to the influence of these very long-duration tracks.
Implications for Data Cleaning:
Given this distribution, here's how to handle the duration_ms field:

Replace Invalid Values: Since you noted that some duration_ms values are -1, indicating missing or incorrect entries, replacing these with the median duration is the most appropriate approach. The median is robust against the influence of the long tail (outliers).}
and i made sure that these were likely to be valid.
---analyzing instrumentalness
to make sure if this analyze was on point and ensures that 0s are valid values in the instrumentalness i asked chatgpt better analyze the effect of output distribution of other features where instrumentalness is 0
{. Instrumentalness Comparison:
The tracks with instrumentalness of 0 form a large subset of the overall data (30% of the entries if we consider 15,001 out of 50,000). This significant proportion suggests that a zero value is common and likely represents tracks with no instrumental content rather than missing data.
2. Other Feature Distributions:
Popularity: The mean popularity of tracks with zero instrumentalness (50.72) is notably higher than the overall average (44.22). This could indicate that non-instrumental tracks tend to be more popular.
Acousticness: These tracks have lower average acousticness (0.213) compared to the overall average (0.306), which aligns with them being less instrumental.
Danceability and Energy: Tracks with zero instrumentalness are more danceable (mean 0.621 vs. 0.558) and have higher energy (mean 0.668 vs. 0.600) than the average, supporting the idea that these tracks are likely more vocal and rhythm-focused.
3. Loudness and Speechiness:
Loudness: Tracks with zero instrumentalness are louder on average (-6.61 dB compared to -9.13 dB overall), which is typical for more commercial, vocal-heavy music.
Speechiness: There's also a higher degree of speechiness (0.135 vs. 0.094) in these tracks, which could be associated with genres that feature more spoken words, like hip-hop or rap.}

---analyzing missing tempo values
With tempo values missing in 5k rows and marked with '?', making the column type as object:

Replace '?' with a numerical placeholder (like NaN) and then convert the column to float.
Impute missing values with a statistical measure (mean or median).

Data Distribution and Imputation Strategy:
The distribution is multimodal, with several key peaks which confirm the presence of groups of songs with similar tempos, often characteristic of specific musical genres or styles.
This distribution supports a genre-specific imputation strategy for missing values, as different genres might have different typical tempos.
For general analysis, imputing missing tempo values using the median or mode of the entire dataset might obscure these nuanced differences between musical styles.

using the genre-specific mode 
This approach allows for more nuanced imputation where missing tempo values are filled in a way that respects the typical tempo characteristics of each genre, 
which could lead to more accurate and meaningful analysis, especially when the tempo could significantly vary between genres.
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#####Feature Engineering######

2 new featureas are created - interaction features
Importance of These Features:
Energy and Danceability: The product of these features can help machine learning models recognize patterns where both attributes are important for classifying certain music genres.
Loudness and Energy: This interaction can identify genres where loudness coupled with high energy defines the music style, such as in hard rock, where loud, energetic music is a hallmark.

1 new feature - aggregate feature-
acoustic_instrumental_ratio
High Values: A high ratio indicates that a track is significantly more acoustic than instrumental. This might be characteristic of genres like folk or acoustic blues.
Low Values: Conversely, a low ratio (closer to zero) would suggest that a track is more instrumental, which might be common in genres like electronic or classical music.

2 categories from the existing ones- tempo_category and duration category
--

---Added polynomial 
added polynomial features to the dataset i do not know if it they are useful or not but i will keep them for now, i do not actually know what they do and their purposes might delete them all but 
first they need analysis,i will decide later whether to keep them or delete them and just use the other newly created features or the original ones.


####### Step 3: Exploratory Data Analysis (EDA)

