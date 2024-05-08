BISMILLAHIRRAHMANIRRAHIM
__________________________
Prediction of music genre
Classify music into genres

The full list of genres included in the CSV are 'Electronic', 'Anime', 'Jazz', 'Alternative', 'Country', 'Rap', 'Blues', 'Rock', 'Classical', 'Hip-Hop'.

https://www.kaggle.com/datasets/vicsuperman/prediction-of-music-genre

1-some songs are in two separate rows for being in two different genres(e.g-Blockhead-Cheer up, You're Not Dead Yet" is both in electronic and jazz)-total 7k
 (also their popularity are different in each instance as well as their genres)
Total songs with multiple genres: 3633

2-2,5k of the songs has empty_field as artist_name
3-5 rows are empty
4-all the song names are valid but some of them are in different Fonts 
5-some of the acousticness is in scientific notation(e.g 4.18e-05) but there is no problem for them they are directly read 
6-no problem seen with danceability
7-some duration_ms is -1 meaning empty cell--5k
8-some instrumentalness values are 0(missing data)-15K
9-some tempo values are missing ? -5K
10-from every label(genre) 5k instances
11-obtained date can be dropped it is probably about on which date the data about that instance is gathered
12-valence means how likely is the song to be positive higher the value positiver the song