#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 16:40:27 2024

@author: tshepo
"""

import pandas as pd
import numpy as np
from collections import Counter


#read data from csv
df=pd.read_csv("movie_dataset.csv")


# Rename the column names only if there is a space in the column name
print("Columns with spaces in the name renamed")
new_col_names = [col.replace(' ', '_') if ' ' in col else col for col in df.columns]
df.columns = new_col_names
# Checking for columns with missing data
missing_data = df.isnull().sum()
# Filtering to show only columns that have missing data
columns_with_missing_data = missing_data[missing_data > 0]
print("Columns with Missing Data:")
print(columns_with_missing_data)
# cleaned data will only be used were columns with missing data are invloved
# Dropping rows with no values
df_cleaned = df.dropna()


print("QUESTION 1")
# Finding the movie with the highest rating
highest_rated_movie = df[df['Rating'] == df['Rating'].max()]
# Displaying the title and rating of the highest rated movie
highest_rated_movie_title = highest_rated_movie['Title'].iloc[0]
highest_rated_movie_rating = highest_rated_movie['Rating'].iloc[0]
print(f"The highest rated movie is '{highest_rated_movie_title}' with a rating of {highest_rated_movie_rating}.")

print("QUESTION 2")
# Calculating the average revenue of all movies
average_revenue = df_cleaned['Revenue_(Millions)'].mean()
print(f"The average revenue of all movies in the dataset is: ${average_revenue:.2f} million")

print("QUESTION 3")
# Filtering the movies from 2015 to 2017
movies_2015_to_2017 = df_cleaned[(df_cleaned['Year'] >= 2015) & (df_cleaned['Year'] <= 2017)]
# Calculating the average revenue for these movies
average_revenue_2015_to_2017 = movies_2015_to_2017['Revenue_(Millions)'].mean()
print(f"The average revenue of movies from 2015 to 2017 is: ${average_revenue_2015_to_2017:.2f} million")

print("QUESTION 4")
# Counting the number of movies released in 2016 using original data
movies_in_2016_count = df[df['Year'] == 2016].shape[0]
print(f"Number of movies released in 2016 on original data: {movies_in_2016_count}")


print("QUESTION 5")
# Counting movies directed by Christopher Nolan using original data
nolan_movies_count = df[df['Director'] == 'Christopher Nolan'].shape[0]
print(f"Number of movies directed by Christopher Nolan: {nolan_movies_count}")

print("QUESTION 6")
# Counting the number of movies with a rating of at least 8.0 original data
highly_rated_movies_count = df[df['Rating'] >= 8.0].shape[0]
print(f"Number of movies with a rating of at least 8.0: {highly_rated_movies_count}")

print("QUESTION 7")
# Calculating the median rating of movies directed by Christopher Nolan original data
nolan_median_rating = df[df['Director'] == 'Christopher Nolan']['Rating'].median()
print(f"The median rating of movies directed by Christopher Nolan is: {nolan_median_rating}")

print("QUESTION 8")
# Grouping the dataset by year and calculating the average rating for each year using original data
yearly_average_rating = df.groupby('Year')['Rating'].mean()
# Finding the year with the highest average rating
year_with_highest_avg_rating = yearly_average_rating.idxmax()
highest_avg_rating = yearly_average_rating.max()
print(f"The year with the highest average rating is {year_with_highest_avg_rating} with an average rating of {highest_avg_rating:.2f}.")

print("QUESTION 9")
# Counting movies made in 2006 using original data
movies_in_2006 = df[df['Year'] == 2006].shape[0]
# Counting movies made in 2016 using original data
movies_in_2016 = df[df['Year'] == 2016].shape[0]
# Calculating the percentage increase
percentage_increase = ((movies_in_2016 - movies_in_2006) / movies_in_2006) * 100
print(f"Percentage increase in the number of movies from 2006 to 2016 using the equation (movies_in_2016 - movies_in_2006) / movies_in_2006) * 100: {percentage_increase:.2f}%")


print("QUESTION 10")
# Split actors into individual names, accounting for potential spaces using original data
actor_series = df['Actors'].str.split(', ').explode().str.strip()
# Count the occurrences of each actor
actor_counts = Counter(actor_series)
# Identify the most common actor
most_common_actor = actor_counts.most_common(1)[0]
print(f"The most common actor is {most_common_actor[0]} with {most_common_actor[1]} appearances.")

print("QUESTION 11")
# Splitting the genres for each movie into individual genres using original data
all_genres = df['Genre'].str.split(',').explode()
# Creating a set of unique genres
unique_genres = set(all_genres)
# Counting the number of unique genres
number_of_unique_genres = len(unique_genres)
print(f"There are {number_of_unique_genres} unique genres in the dataset.")


print("QUESTION 12")
# Calculating the correlation matrix using cleaned data since revenue is part of analysis
correlation_matrix = df_cleaned.select_dtypes(include=[np.number]).corr()
# Displaying the correlation matrix
print(correlation_matrix)

