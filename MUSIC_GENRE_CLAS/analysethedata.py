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
print(data.iloc[0:5, 0:10])

#print the rest of the columns
print(data.iloc[0:5, 10:20])