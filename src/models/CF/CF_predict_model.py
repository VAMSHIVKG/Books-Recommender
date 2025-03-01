INPUT = r'models\CF\U_I_M.pkl'
MODEL = r'models\CF\FInal_model.pkl'
BOOKS = r'models\CF\cleaned_books.csv'

import pandas as pd
import joblib
import numpy as np
import pickle

user_item_matrix = joblib.load(INPUT)
books_df = pd.read_csv(BOOKS)

with open(MODEL, 'rb') as f:
    model = pice.load(f)

model = job #
lllib.load(MODEL)

user_item_matrix = user_item_mrix.fillna(0)

print(user_item_matrix)

def get_CF_n_recommendations(user_id ,n=30 ):

    # Get the books the user has not rated (assuming ratings are on a scale from 0 to 10)
    unrated_books = user_item_matrix.loc[user_id][user_item_matrix.loc[user_id] == 0].index

    # Predict ratings for unrated books
    predictions = []
    for book_id in unrated_books:
        prediction = model.predict(user_id, book_id)
        predictions.append((book_id, prediction.est))

    # Sort the predictions in descending order
    predictions.sort(key=lambda x: x[1], reverse=True)

    # Display the top N recommendations (adjust N as needed)
    top_n_recommendations = predictions[:n]

    df = pd.DataFrame(columns=['ISBN', 'Title', 'Author', 'Publication_year', 'Publisher',
                               'Book_image', 'pred_rating'])
    for i in range(len(top_n_recommendations)):
        row = books_df[books_df['Title'] == top_n_recommendations[i][0]]
        row['pred_rating'] = top_n_recommendations[i][1]
        df = pd.concat([row, df])

    return df.sort_values(by='pred_rating', ascending=False)


def get_random_user_id():

    x = user_item_matrix.index
    User_id_df = pd.DataFrame(x)
    return User_id_df.mple(1).values[0][0]

print(get_random_user_id())
print(get_CF_n_recommendations(1234))

########################################################################################

def get_user_profile(user_id):
    Readed_books = pd.DataFrame(
        user_item_matrix.loc[user_id][user_item_matrix.loc[user_id] > 0]).reset_index()
    df = pd.DataFrame(columns=['ISBN', 'Title', 'Author', 'Publication_year', 'Publisher',
                               'Book_image', 'Usrt_rating'])
    for i in range(len(Readed_books['Title'])):
        row = books_df[books_df['Title'] == Readed_books['Title'][i]]
        row['Usrt_rating'] = Readed_books[user_id][i]
        df = pd.concat([row, d)

    return df.sort_values(by='Usrt_rating', ascending=False)

print(get_user_profile(Random_User_id))