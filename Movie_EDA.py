import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the dataset
df = pd.read_csv("cleaned_movie_dataset.csv")

# Show first 5 rows

print(df.head())

# Basic dataset info

print(df.shape)

# Dataset has 9985 rows and 9 columns

print(df.info())
# All columns have non-null values (no missing data)
# Numerical columns: popularity, vote_average, vote_count
# Categorical columns: title, genre, original_language, overview, release_date

print(df.describe())
# Average movie rating (vote_average) is around 6.6
# Most movies have vote_count less than 1500
# Popularity values vary a lot (some movies are extremely popular)

#===========================================================================================


# EDA: Vote Average Analysis

plt.hist(df["vote_average"])
plt.xlabel("Vote Average")
plt.ylabel("Count")
plt.title("Distribution of Movie Ratings")
plt.show()


# Most movies are rated between 6 and 7
# Very few movies have ratings above 8

#===========================================================================================


# popularity

plt.hist(df["popularity"])
plt.xlabel("Popularity")
plt.ylabel("Count")
plt.title("Distribution of Movie Popularity")
plt.show()

# Most movies have low popularity
# Only a few movies are extremely popular (outliers)

#===========================================================================================

# Do popular movies always have high ratings?

plt.scatter(df["popularity"], df["vote_average"])
plt.xlabel("Popularity")
plt.ylabel("Vote Average")
plt.title("Popularity vs Movie Rating")
plt.show()

# High popularity does not always mean high rating
# There is no strong correlation between popularity and vote_average




# =================================================================================


# EDA: Original Language Analysis

avg_rating_per_language = df.groupby('original_language')['vote_average'].mean().sort_values(ascending=False)
print(avg_rating_per_language.head(10))

avg_rating_per_language.head(10).plot(kind='bar', figsize=(12,6))
plt.xlabel("Language")
plt.ylabel("Average Rating")
plt.title("Top 10 Languages by Average Movie Rating")
plt.show()

# Movies in some less common languages tend to have higher average ratings
# English movies have a wide range of ratings
# Language alone does not determine how highly a movie is rated


#=================================================================================


# EDA: Release Year Analysis


df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
df['release_year'] = df['release_date'].dt.year

movies_per_year = df['release_year'].value_counts().sort_index()

plt.figure(figsize=(15,6))
movies_per_year.plot(kind='line')
plt.xlabel("Year")
plt.ylabel("Number of Movies")
plt.title("Number of Movies Released Each Year")
plt.show()

# Movie production increased over time
# Some years have fewer movies, possibly due to missing data or events


#===========================================================================================


# EDA: Top Rated and Most Popular Movies


top_rated = df.sort_values(by='vote_average', ascending=False).head(10)
print(top_rated[['title', 'vote_average', 'popularity']])

most_popular = df.sort_values(by='popularity', ascending=False).head(10)
print(most_popular[['title', 'popularity', 'vote_average']])

# Some top-rated movies are not very popular
# Some most popular movies have only average ratings

#====================================================================================

# EDA: Average Rating per Genre

avg_rating_per_genre = df.groupby('genre')['vote_average'].mean().sort_values(ascending=False)
print(avg_rating_per_genre)

avg_rating_per_genre.plot(kind='bar', figsize=(12,6))
plt.xlabel("Genre")
plt.ylabel("Average Rating")
plt.title("Average Movie Rating per Genre")
plt.show()

# Some genres like Documentary or Animation have higher average ratings
# Action and Horror usually have lower average ratings


