import numpy
import pandas
import os


#print current working directory
#print(os.getcwd())

# Load the dataset
data = pandas.read_csv('./MUSIC_GENRE_CLAS/music_genre.csv')

# 1. Print the first 5 rows of the dataset
#print(data.head()) -- the output can not show the whole columns

# print the first 5 rows of the dataset with the first 10 columns
#print(data.iloc[49500:49505, 0:10])

#print the rest of the columns
#print(data.iloc[49500:49505, 10:20])

#print the instances with same artist_name and track_name , same ones under each other
#print(data[data.duplicated(['artist_name', 'track_name'], keep=False)].sort_values('artist_name'))



# drop the empty rows
data = data.dropna()

#print(data[data.duplicated(['artist_name', 'track_name'], keep=False)].sort_values('artist_name')) #7k
#save the duplicate data above to a csv file
#data[data.duplicated(['artist_name', 'track_name'], keep=False)].sort_values('artist_name').to_csv('./MUSIC_GENRE_CLAS/duplicate.csv')

#print the artist_name where it is empty_field
#print(data[data['artist_name'] == 'empty_field']) #2,5k 
#print where the tracck_name is empty_field
#print(data[data['track_name'] == 'empty_field']) 
#print where some of the acousticness is in scientific notation(e.g 4.18E-05)
#print where acousticness is int scientific notation
#print where instance id is 49375
#print where duration_ms is -1
#print(data[data['duration_ms'] == -1]) #5k
#print where instrumentalness is 0
#print(data[data['instrumentalness'] == 0]) 
#print where tempo is ?
#print(data[data['tempo'] == '?']) 
#find the track_name s that is both in music_genre hip-hop and in  rap
#print(data[(data['music_genre'] == 'hip-hop') & (data['music_genre'] == 'rap')]) #0

# Assuming the column containing genres is named 'genre'
#hip_hop_rap_songs = data[data['music_genre'].str.contains('hip-hop') & data['music_genre'].str.contains('rap')]

# Print the result
#print(hip_hop_rap_songs)

#bring any song which is in more than one genre sort them together ,they are not coma separated but separate rows
# # Group by the song name and aggregate genres into a list
# grouped = data.groupby('track_name')['music_genre'].agg(list).reset_index()
# # Filter songs that have more than one genre
# multiple_genres = grouped[grouped['music_genre'].apply(lambda x: len(set(x)) > 1)]
# for index, row in multiple_genres.iterrows():
#     print(f"Song: {row['track_name']}")
#     for genre in set(row['music_genre']):  # Using set to avoid duplicate genres
#         print(f"  Genre: {genre}")
#     print("\n")  # Adds a space between different songs

# Group by the song name and artist name, and aggregate genres into a list
grouped = data.groupby(['track_name', 'artist_name'])['music_genre'].agg(list).reset_index()
# Filter songs that have more than one genre
multiple_genres = grouped[grouped['music_genre'].apply(lambda x: len(set(x)) > 1)]
count = 0
for index, row in multiple_genres.iterrows():
    print(f"Song: {row['track_name']}, Artist: {row['artist_name']}")
    count += 1
    for genre in set(row['music_genre']):  # Using set to avoid duplicate genres
        print(f"  Genre: {genre}")
    print("\n")  # Adds a space between different songs

print(f"Total songs with multiple genres: {count}")



















