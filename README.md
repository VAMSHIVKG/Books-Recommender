# Hybrid Book Recommendation System

This academic project implements a Hybrid Book Recommendation System using Collaborative Filtering and Content-Based Filtering. It provides users with top-rated books and personalized recommendations based on their customer ID.

As the team lead, I guided the project from development to deployment, ensuring efficient recommendation generation and a user-friendly interface.

## 📌 Overview

The Hybrid Book Recommendation System is an intelligent system that suggests books to users based on a combination of collaborative filtering, content-based filtering, and metadata-based approaches. This project aims to enhance book discovery by providing personalized recommendations based on user preferences and book similarities.

## 🚀 Features

- 📚 Hybrid Recommendation: Combines content-based filtering and collaborative filtering for more accurate results.

- 🔍 Search Functionality: Users can search for books by title, author, or genre.

- 💡 Personalized Suggestions: Recommends books based on user interactions and ratings.

- 📊 Similarity Metrics: Uses cosine similarity and other techniques to determine book relevance.

- 🛠 Error Handling: Provides meaningful error messages for invalid queries.

- 🌐 Interactive UI: Built using Streamlit for an easy-to-use interface.

## 🏗️ Technology Stack

**Programming Language:** Python

**Libraries Used:**

**pandas** - Data handling and preprocessing

**numpy** - Numerical operations

**scikit-learn** - Machine learning models and similarity computations

**Streamlit** - Web framework for building an interactive UI

**NLTK** - Text processing for content-based filtering

**Database:** CSV

## 📌 Installation & Setup

### Prerequisites

- Python (>=3.8)

- Virtual Environment (optional but recommended)

- Required libraries (can be installed using the requirements.txt file)

### Steps

1. Clone the repository:

```bash
  git clone https://github.com/yourusername/hybrid-book-recommender.git
  cd hybrid-book-recommender
```

2. Install dependencies:

```bash
  pip install -r requirements.txt
```

3. Run the application:

```bash
  streamlit run app.py
```

4. Open in browser:

```bash
  http://localhost:8501
```

## 📊 How It Works

1. **Data Collection**: The system processes book metadata, user ratings, and reviews.

2. **Content-Based Filtering**: Analyzes book descriptions and genres using cosine similarity to find similar books.

3. **Collaborative Filtering**: Uses matrix factorization techniques to recommend books based on user preferences.

4. **Hybrid Approach**: Combines both methods for more precise recommendations.

## 📂 Project Structure

📁 hybrid-book-recommender/

    │── 📄 README.md
    │── 📄 requirements.txt
    │── 📂 data/ (Contains book metadata & user interactions)
    │── 📂 models/ (Pre-trained ML models & similarity matrices)
    │── 📂 src/ (Main source code & implementation)
    │── 📄 app.py (Entry point for the application using Streamlit)

## 💡 Future Enhancements

✅ Deploy as a cloud-based service using AWS/GCP.

✅ Enhance NLP models for better content-based filtering.

✅ Implement deep learning for recommendation improvement.

✅ Add user authentication for personalized recommendations.

## 👥 Contributors

- **Vamshi Krishna(Me)** (Lead Project Developer & Researcher)

- **Subhash** and **Bogeswar** (Members)

- **Dr. P L Srinivas Murthy - Professor** (Mentor)
