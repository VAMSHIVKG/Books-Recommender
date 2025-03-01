'''import re, pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import nltk
nltk.download('stopwords')
nltk.download('wordnet')


from flask_cors import CORS
from flask import Flask, jsonify, request

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer

from sklearn.metrics.pairwise import cosine_similarity

import tensorflow as tf
from keras.layers import LSTM
from tensorflow.keras.models import load_model

# Ensure that custom objects are handled
Fmodel = load_model(model_weights, custom_objects={'LSTM': LSTM})

# Continue with the rest of your code



app = Flask(__name__)
CORS(app)
##################################################################################################################

tokenizer_path = 'weights/TOKENIZER.pkl'
model_weights = 'weights/DESCRIPTION_SIMILARITY.h5'
book_features_file = 'weights/BOOK_FEATURES.npz'

max_length = 300

lemmatizer = WordNetLemmatizer()
re_tokenizer = RegexpTokenizer(r'\w+')
stopwords_list = stopwords.words('english')

with open(tokenizer_path, 'rb') as fp:
    tokenizer = pickle.load(fp)

Fmodel = tf.keras.models.load_model(model_weights)

book_ids, book_features = np.load(book_features_file, allow_pickle=True).values()

##################################################################################################################

def load_data():
    # Read using MongoDB instead of CSV if you need
    df_book_metadata = pd.read_csv('data/books_v2.csv')
    df_user = pd.read_csv('data/users.csv')

    df_book_metadata = df_book_metadata.dropna(subset=['description'])
    df_book_metadata = df_book_metadata[:2000]
    bookIds = df_book_metadata['bookId'].values
    titles = df_book_metadata['title'].values
    authors = df_book_metadata['author'].values
    coverImgs = df_book_metadata['coverImg'].values
    ratings = df_book_metadata['rating'].values

    userIds = df_user['userId'].values
    bookIds = df_user['bookId'].values
    locations = df_user['location'].values
    Occupied = df_user['Occupied'].values



    book_metadata = {bookId : {
                    'title': title, 
                    'author': author, 
                    'coverImg': coverImg, 
                    'rating': rating
                    } for bookId, title, author, coverImg, rating in zip(bookIds, titles, authors, coverImgs, ratings)}
                    
    user_metadata = {bookId : {
                        'userId': userId, 
                        'location': location, 
                        'Occupied': Occupied
                        } for userId, bookId, location, Occupied in zip(userIds, bookIds, locations, Occupied)} 
    
    return book_metadata, user_metadata

def lemmatization(lemmatizer,sentence):
    lem = [lemmatizer.lemmatize(k) for k in sentence]
    return [k for k in lem if k]

def remove_stop_words(stopwords_list,sentence):
    return [k for k in sentence if k not in stopwords_list]

def preprocess_one(description):
    description = description.lower()
    remove_punc = re_tokenizer.tokenize(description) # Remove puntuations
    remove_num = [re.sub('[0-9]', '', i) for i in remove_punc] # Remove Numbers
    remove_num = [i for i in remove_num if len(i)>0] # Remove empty strings
    lemmatized = lemmatization(lemmatizer,remove_num) # Word Lemmatization
    remove_stop = remove_stop_words(stopwords_list,lemmatized) # remove stop words
    updated_description = ' '.join(remove_stop)
    return updated_description

def preprocessed_data(descriptions):
    updated_descriptions = []
    if isinstance(descriptions, np.ndarray) or isinstance(descriptions, list):
        updated_descriptions = [preprocess_one(description) for description in descriptions]
    elif isinstance(descriptions, np.str_)  or isinstance(descriptions, str):
        updated_descriptions = [preprocess_one(descriptions)]

    return np.array(updated_descriptions)


def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371 # metres
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2-lat1)
    delta_lambda = np.radians(lon2-lon1)

    a = np.sin(delta_phi/2) * np.sin(delta_phi/2) + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda/2) * np.sin(delta_lambda/2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

    d = R * c
    return d


book_metadata, user_metadata = load_data()

################################################ Inference Function ###############################################
def extract_similar_books(
                        description, 
                        user_location,
                        max_distance=50,
                        top_n=30
                        ):
    description = preprocessed_data(description)
    d_seq = tokenizer.texts_to_sequences(description)
    d_pad = tf.keras.preprocessing.sequence.pad_sequences(
                                                        d_seq,
                                                        maxlen=max_length,
                                                        padding='pre',
                                                        truncating='pre'
                                                        )
    fpred = Fmodel.predict(d_pad)
    fpred = fpred.reshape(1, -1)
    sim = cosine_similarity(fpred, book_features)
    sim = sim.reshape(-1)
    sim = np.argsort(sim)[::-1]
    sim_book_ids =  book_ids[sim[:top_n]]
    sim_book_data = {}
    for book_id in sim_book_ids:
        try:
            book_rating = int(book_metadata[book_id]['rating'])
            rec_book_location = eval(user_metadata[book_id]['location'])
            rec_book_occupied = int(user_metadata[book_id]['Occupied'])
            distance = haversine_distance(user_location[0], user_location[1], rec_book_location[0], rec_book_location[1])
            distance = round(distance, 2)
            if (book_rating >= 3) and (distance <= max_distance) and (rec_book_occupied == 0):
                sim_book_data[book_id] = book_metadata[book_id] 
                sim_book_data[book_id]['distance'] = distance
                sim_book_data[book_id]['rating'] = book_rating
        except:
            continue

    sim_book_data_reponse = []
    for book_id, book_data in sim_book_data.items():
        sim_book_data_reponse.append({
                                    "bookId": f"{book_id}",
                                    "title": f"{book_data['title']}",
                                    "author": f"{book_data['author']}",
                                    "coverImg": f"{book_data['coverImg']}",
                                    "rating": f"{book_data['rating']}",
                                    "distance": f"{book_data['distance']} Km"
                                    })


    return sim_book_data_reponse


@app.route("/rec", methods=["POST"])
def rec():
    if request.method == "POST":
        data = request.get_json()
        description = data['description']
        user_location = eval(data['user_location'])
        max_distance = int(data['max_distance']) if 'max_distance' in data else 50
        top_n = int(data['top_n']) if 'top_n' in data else 30

        sim_book_data_reponse = extract_similar_books(
                                                    description, 
                                                    user_location,
                                                    max_distance,
                                                    top_n
                                                    )
        return jsonify(sim_book_data_reponse)
    
if __name__ == "__main__": 
    app.run(
            debug=True, 
            host='0.0.0.0', 
            port=5000, 
            threaded=False
            )
    
'''

{
    "description" : "WINNING MEANS FAME AND FORTUNE.LOSING MEANS CERTAIN DEATH.THE HUNGER GAMES HAVE BEGUN. . . .In the ruins of a place once known as North America lies the nation of Panem, a shining Capitol surrounded by twelve outlying districts. The Capitol is harsh and cruel and keeps the districts in line by forcing them all to send one boy and once girl between the ages of twelve and eighteen to participate in the annual Hunger Games, a fight to the death on live TV.Sixteen-year-old Katniss Everdeen regards it as a death sentence when she steps forward to take her sister's place in the Games. But Katniss has been close to dead beforeâ€”and survival, for her, is second nature. Without really meaning to, she becomes a contender. But if she is to win, she will have to start making choices that weight survival against humanity and life against love.",
    "user_location" : "[6.9271, 79.8612]",
}

''''''
import re
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf

from flask_cors import CORS
from flask import Flask, jsonify, request

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer

from sklearn.metrics.pairwise import cosine_similarity
from keras.layers import LSTM
from tensorflow.keras.models import load_model

app = Flask(__name__)
CORS(app)

##################################################################################################################

tokenizer_path = 'weights/TOKENIZER.pkl'
model_weights = 'weights/DESCRIPTION_SIMILARITY.h5'
book_features_file = 'weights/BOOK_FEATURES.npz'

max_length = 300

lemmatizer = WordNetLemmatizer()
re_tokenizer = RegexpTokenizer(r'\w+')
stopwords_list = stopwords.words('english')

with open(tokenizer_path, 'rb') as fp:
    tokenizer = pickle.load(fp)

# Use custom_objects parameter to handle custom layers like LSTM
Fmodel = load_model(model_weights, custom_objects={'LSTM': LSTM})

book_ids, book_features = np.load(book_features_file, allow_pickle=True).values()

##################################################################################################################

def load_data():
    df_book_metadata = pd.read_csv('data/books_v2.csv')
    df_user = pd.read_csv('data/users.csv')

    df_book_metadata = df_book_metadata.dropna(subset=['description'])
    df_book_metadata = df_book_metadata[:2000]
    bookIds = df_book_metadata['bookId'].values
    titles = df_book_metadata['title'].values
    authors = df_book_metadata['author'].values
    coverImgs = df_book_metadata['coverImg'].values
    ratings = df_book_metadata['rating'].values

    userIds = df_user['userId'].values
    bookIds_user = df_user['bookId'].values
    locations = df_user['location'].values
    Occupied = df_user['Occupied'].values

    book_metadata = {bookId: {
        'title': title,
        'author': author,
        'coverImg': coverImg,
        'rating': rating
    } for bookId, title, author, coverImg, rating in zip(bookIds, titles, authors, coverImgs, ratings)}

    user_metadata = {bookId: {
        'userId': userId,
        'location': location,
        'Occupied': Occupied
    } for userId, bookId, location, Occupied in zip(userIds, bookIds_user, locations, Occupied)}

    return book_metadata, user_metadata

def lemmatization(lemmatizer, sentence):
    lem = [lemmatizer.lemmatize(k) for k in sentence]
    return [k for k in lem if k]

def remove_stop_words(stopwords_list, sentence):
    return [k for k in sentence if k not in stopwords_list]

def preprocess_one(description):
    description = description.lower()
    remove_punc = re_tokenizer.tokenize(description)  # Remove punctuation
    remove_num = [re.sub('[0-9]', '', i) for i in remove_punc]  # Remove Numbers
    remove_num = [i for i in remove_num if len(i) > 0]  # Remove empty strings
    lemmatized = lemmatization(lemmatizer, remove_num)  # Word Lemmatization
    remove_stop = remove_stop_words(stopwords_list, lemmatized)  # Remove stop words
    updated_description = ' '.join(remove_stop)
    return updated_description

def preprocessed_data(descriptions):
    if isinstance(descriptions, (np.ndarray, list)):
        updated_descriptions = [preprocess_one(description) for description in descriptions]
    elif isinstance(descriptions, (np.str_, str)):
        updated_descriptions = [preprocess_one(descriptions)]

    return np.array(updated_descriptions)

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371  # metres
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)

    a = np.sin(delta_phi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    d = R * c
    return d

book_metadata, user_metadata = load_data()

################################################ Inference Function ###############################################

def extract_similar_books(description, user_location, max_distance=50, top_n=30):
    description = preprocessed_data(description)
    d_seq = tokenizer.texts_to_sequences(description)
    d_pad = tf.keras.preprocessing.sequence.pad_sequences(
        d_seq,
        maxlen=max_length,
        padding='pre',
        truncating='pre'
    )
    fpred = Fmodel.predict(d_pad)
    fpred = fpred.reshape(1, -1)
    sim = cosine_similarity(fpred, book_features)
    sim = sim.reshape(-1)
    sim = np.argsort(sim)[::-1]
    sim_book_ids = book_ids[sim[:top_n]]
    sim_book_data = {}
    for book_id in sim_book_ids:
        try:
            book_rating = int(book_metadata[book_id]['rating'])
            rec_book_location = eval(user_metadata[book_id]['location'])
            rec_book_occupied = int(user_metadata[book_id]['Occupied'])
            distance = haversine_distance(user_location[0], user_location[1], rec_book_location[0], rec_book_location[1])
            distance = round(distance, 2)
            if (book_rating >= 3) and (distance <= max_distance) and (rec_book_occupied == 0):
                sim_book_data[book_id] = book_metadata[book_id]
                sim_book_data[book_id]['distance'] = distance
                sim_book_data[book_id]['rating'] = book_rating
        except:
            continue

    sim_book_data_response = []
    for book_id, book_data in sim_book_data.items():
        sim_book_data_response.append({
            "bookId": f"{book_id}",
            "title": f"{book_data['title']}",
            "author": f"{book_data['author']}",
            "coverImg": f"{book_data['coverImg']}",
            "rating": f"{book_data['rating']}",
            "distance": f"{book_data['distance']} Km"
        })

    return sim_book_data_response

@app.route("/rec", methods=["POST"])
def rec():
    if request.method == "POST":
        data = request.get_json()
        description = data['description']
        user_location = eval(data['user_location'])
        max_distance = int(data['max_distance']) if 'max_distance' in data else 50
        top_n = int(data['top_n']) if 'top_n' in data else 30

        sim_book_data_response = extract_similar_books(
            description,
            user_location,
            max_distance,
            top_n
        )
        return jsonify(sim_book_data_response)

if __name__ == "__main__":
    app.run(
        debug=True,
        host='0.0.0.0',
        port=5000,
        threaded=False
    )
