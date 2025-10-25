"""
Build a Book Recommender System

Recommender systems are used in all sorts of organizations to help users make decisions and, for many companies, earn more revenue. Prototyping simple recommender systems also does not need to take a lot of time. In this project, we will build a book recommender system for Books’R’Us using Surprise.

Books’R’Us is a national bookstore chain that sells books of all sorts to people all over the country. They recently have built their website, and now want to add a book recommender system to their site. We will prepare and train the recommender system using book review data left on their site. This data has been put together in a Pandas DataFrame called book_ratings.

"""

import pandas as pd
import codecademylib3
from surprise import Reader

book_ratings = pd.read_csv('goodreads_ratings.csv')
print(book_ratings.head())

#1. Print dataset size and examine column data types
print(len(book_ratings))
print(book_ratings.info())

#2. Distribution of ratings
print(book_ratings['rating'].value_counts())

#3. Filter ratings that are out of range
book_ratings = book_ratings[book_ratings['rating']!=0]

#4. Prepare data for surprise: build a Suprise reader object
from surprise import Reader
reader = Reader(rating_scale=(1, 5))

#5. Load `book_ratings` into a Surprise Dataset
from surprise import Dataset
rec_data = Dataset.load_from_df(book_ratings[['user_id',
                                              'book_id',
                                              'rating']],
                                reader)

#6. Create a 80:20 train-test split and set the random state to 7
from surprise.model_selection import train_test_split
trainset, testset = train_test_split(rec_data, test_size=.2, random_state=7)

#7. Use KNNBasice from Surprise to train a collaborative filter
from surprise import KNNBasic
nn_algo = KNNBasic()
nn_algo.fit(trainset)

#8. Evaluate the recommender system
from surprise import accuracy
predictions = nn_algo.test(testset)
accuracy.rmse(predictions)

#9. Prediction on a user who gave the "The Three-Body Problem" a rating of 5
print(nn_algo.predict('8842281e1d1347389f2ab93d60773d4d', '18007564').est)