import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
# import json
# import csv

# moveid_and_tv_df = pd.read_json("Movies_and_TV.json", lines=True)
# Desktop/Grad School 2022-2023/Spring 2023/CSE 272/Assignment2
# Recommendation_Systems.py

# Load data
dataframe = pd.read_json('Magazine_Subscriptions.json', lines=True)

# Drop unnecessary columns
dataframe = dataframe.drop(columns=['reviewTime', 'reviewText', 'verified', 'style', 'vote', 'reviewerName', 'unixReviewTime', 'summary'])

# Split data into training and testing sets
train_data, test_data = train_test_split(dataframe, test_size=0.2)

# Prepare training data
training_pivot = train_data.pivot_table(values='overall', index='reviewerID', columns='asin').fillna(0)

# Generate pivot table
pivot_table = training_pivot.apply(np.sign)
# print(pivot_table)

# Gradient Descent
item_sim = np.dot(pivot_table.T.values, pivot_table.T.values.T)

predictions = np.random.rand(*pivot_table.T.values.shape)
learning_rate = 0.001
num_iter = 100

for _ in range(num_iter):
    predictions += learning_rate * (item_sim.T.dot(pivot_table.T.values) / np.array([np.abs(item_sim.T).sum(axis=1)]).T)

original_ratings = pivot_table.T.values

# MAE
non_zero_indices = np.nonzero(original_ratings)
prediction_non_zero = predictions[non_zero_indices].flatten()
original_ratings_non_zero = original_ratings[non_zero_indices].flatten()
absolute_errors = np.abs(prediction_non_zero - original_ratings_non_zero)
mae_score =  np.mean(absolute_errors)

# RMSE
non_zero_indices = np.nonzero(original_ratings)
prediction_non_zero = predictions[non_zero_indices].flatten()
original_ratings_non_zero = original_ratings[non_zero_indices].flatten()
squared_errors = np.square(prediction_non_zero - original_ratings_non_zero)
mean_squared_error = np.mean(squared_errors)
rmse_score = np.sqrt(mean_squared_error)

# Print results
print("Mean Absolute Error:", mae_score)
print("Root Mean Square Error:", rmse_score)

