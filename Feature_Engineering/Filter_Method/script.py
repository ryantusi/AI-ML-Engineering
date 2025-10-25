"""
Transforming Data into Features

You are a data scientist at a clothing company and are working with a data set of customer reviews. This dataset is originally from Kaggle and has a lot of potential for various machine learning purposes. You are tasked with transforming some of these features to make the data more useful for analysis. To do this, you will have time to practice the following:

    Transforming categorical data
    Scaling your data
    Working with date-time features

Let’s get started!
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

#import data
reviews = pd.read_csv("reviews.csv")
 
#print column names
print(reviews.head(10))
 
#print .info
print(reviews.info())

#look at the counts of recommended
print(reviews["recommended"].value_counts())
 
#create binary dictionary
binary_dict = {False:0, True:1}
 
#transform column
reviews["recommended"] = reviews["recommended"].map(binary_dict)
 
#print your transformed column
print(reviews.head(10))

#look at the counts of rating
print(reviews["rating"].value_counts())
 
#create dictionary
rating_dict = {
  "Loved it":5,
  "Liked it":4,
  "Was okay":3,
  "Not great":2,
  "Hated it":1
}
 
#transform rating column
reviews["rating_trans"] = reviews["rating"].map(rating_dict)
 
#print your transformed column values
print(reviews["rating_trans"].value_counts())

#get the number of categories in a feature
print(reviews["department_name"].value_counts())
 
#perform get_dummies
one_hot = pd.get_dummies(reviews["department_name"])
 
#join the new columns back onto the original
reviews = reviews.join(one_hot)

#print column names
print(reviews.columns)

#transform review_date to date-time data
reviews["new_review_date"] = pd.to_datetime(reviews["review_date"])

#print review_date data type 
print(reviews["new_review_date"].dtype)

#get numerical columns
#get numerical columns
reviews = reviews[['clothing_id', 'age', 'recommended', 'rating_trans', 'Bottoms', 'Dresses', 'Intimate', 'Jackets', 'Tops', 'Trend']].copy()
 
#reset index
reviews = reviews.set_index("clothing_id")

#instantiate standard scaler
scaler = StandardScaler()
 
#fit transform data
new_reviews = scaler.fit_transform(reviews)

print(new_reviews[:5])


