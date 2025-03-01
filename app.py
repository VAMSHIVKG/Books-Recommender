# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib

# # Define the paths
# FILTERD_BOOKS = r'C:\Users\gattu\hybrid-book-recommendation-system\models\content base\filtired_books_cb.pkl'
# SIM_MATRIX = r'C:\Users\gattu\hybrid-book-recommendation-system\data\processed\For content base\similarity_matrix_cb.h5'

# # Downcast data types
# def downcast_dtypes(df):
#     float_cols = [c for c in df if df[c].dtype == "float64"]
#     int_cols = [c for c in df if df[c].dtype == "int64"]

#     df[float_cols] = df[float_cols].astype("float32")
#     df[int_cols] = df[int_cols].astype("int32")

#     return df

# # Loading CSV in chunks
# chunksize = 10000
# chunk_list = []

# for chunk in pd.read_csv(FILTERD_BOOKS, chunksize=chunksize):
#     chunk = downcast_dtypes(chunk)
#     chunk_list.append(chunk)

# books_df = pd.concat(chunk_list, axis=0)

# # Load large similarity matrix from HDF5
# with pd.HDFStore(SIM_MATRIX) as store:
#     if 'sim_matrix' in store:
#         sim_matrix = store.get('sim_matrix').values
#     else:
#         raise KeyError("No object named 'sim_matrix' in the file")

# # Function to get top N books
# def get_top_n_books(n):
#     df = books_df.sort_values(by=['Rating_count', 'Avg_rating'], ascending=False).head(n)
#     return df

# # Function to recommend books by ID
# def recommendation_by_id(book_id):
#     try:
#         if book_id >= sim_matrix.shape[0]:
#             raise IndexError(f'Book ID {book_id} is out of bounds for the similarity matrix of size {sim_matrix.shape[0]}')

#         book_similarities = sim_matrix[book_id]
#         similar_indices = np.argsort(book_similarities)[1:11]
#         recom_df = books_df.iloc[similar_indices]
#         return recom_df
#     except KeyError:
#         st.error('Book not found in the similarity matrix. Please choose another book.')
#     except IndexError as e:
#         st.error(e)

# # Streamlit app
# st.title('Book Recommendation System')

# st.header('Top N Books')
# num_books = st.number_input('Enter the number of top books to display:', min_value=1, max_value=100, value=10)
# top_books = get_top_n_books(num_books)
# st.write(top_books)

# st.header('Book Recommendations by ID')
# book_id_input = st.number_input('Enter a book ID for recommendations:', min_value=0, value=0)
# recommendations = recommendation_by_id(book_id_input)
# if recommendations is not None:
#     st.write(recommendations)

# if st.button('Get Recommendations'):
#     recommendations = recommendation_by_id(book_id_input)
#     if recommendations is not None:
#         st.write(recommendations)

# if st.button('Show Top N Books'):
#     top_books = get_top_n_books(num_books)
#     st.write(top_books)

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Define the paths
FILTERD_BOOKS = r'C:\Users\gattu\hybrid-book-recommendation-system\models\content base\filtired_books_cb.pkl'
SIM_MATRIX = r'C:\Users\gattu\hybrid-book-recommendation-system\data\processed\For content base\similarity_matrix_cb.h5'

# Downcast data types
def downcast_dtypes(df):
    float_cols = [c for c in df if df[c].dtype == "float64"]
    int_cols = [c for c in df if df[c].dtype == "int64"]

    df[float_cols] = df[float_cols].astype("float32")
    df[int_cols] = df[int_cols].astype("int32")

    return df

# Loading CSV in chunks
chunksize = 10000
chunk_list = []

for chunk in pd.read_csv(FILTERD_BOOKS, chunksize=chunksize):
    chunk = downcast_dtypes(chunk)
    chunk_list.append(chunk)

books_df = pd.concat(chunk_list, axis=0)

# Load large similarity matrix from HDF5
with pd.HDFStore(SIM_MATRIX) as store:
    if 'sim_matrix' in store:
        sim_matrix = store.get('sim_matrix').values
    else:
        raise KeyError("No object named 'sim_matrix' in the file")

# Function to get top N books
def get_top_n_books(n):
    df = books_df.sort_values(by=['Rating_count', 'Avg_rating'], ascending=False).head(n)
    return df

# Function to recommend books by ID
def recommendation_by_id(book_id):
    try:
        if book_id >= sim_matrix.shape[0]:
            raise IndexError(f'Book ID {book_id} is out of bounds for the similarity matrix of size {sim_matrix.shape[0]}')

        book_similarities = sim_matrix[book_id]
        similar_indices = np.argsort(book_similarities)[1:11]
        recom_df = books_df.iloc[similar_indices]
        return recom_df
    except KeyError:
        st.error('Book not found in the similarity matrix. Please choose another book.')
    except IndexError as e:
        st.error(e)

# Streamlit app
st.title('Book Recommendation System')

num_books = st.number_input('Enter the number of top books to display:', min_value=1, max_value=100, value=10)
st.header('Top '+str(num_books)+' Books')
top_books = get_top_n_books(num_books)

# Displaying top books
st.subheader('Top Books')
for index, row in top_books.iterrows():
    st.image(row['Book_image'], width=100)
    st.write(f"**Title:** {row['Title']}")
    st.write(f"**Author:** {row['Author']}")
    st.write(f"**Rating:** {row['Avg_rating']} ({row['Rating_count']} ratings)")
    # st.write(f"**Description:** {row['Description'][:200]}...")  # Display first 200 characters of description
    st.write('---')

st.header('Book Recommendations by ID')
book_id_input = st.number_input('Enter a book ID for recommendations:', min_value=0, value=0)
if st.button('Get Recommendations'):
    recommendations = recommendation_by_id(book_id_input)
    if recommendations is not None:
        st.subheader('Recommended Books')
        for index, row in recommendations.iterrows():
            st.image(row['Book_image'], width=100)
            st.write(f"**Title:** {row['Title']}")
            st.write(f"**Author:** {row['Author']}")
            st.write(f"**Rating:** {row['Avg_rating']} ({row['Rating_count']} ratings)")
            # st.write(f"**Description:** {row['Description'][:200]}...")
            st.write('---')

