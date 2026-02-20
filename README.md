# 🎬 Movie Recommendation System

A simple Content-Based Movie Recommendation System built using Python and Streamlit.

This project recommends movies similar to a selected movie based on genres, keywords, cast, director, and overview.

---

## 🚀 Tech Stack

- Python
- Pandas
- Scikit-learn
- Streamlit

---

## 📂 Files

- `app.py` → Streamlit web app
- `movie_recommender.py` → Model building script
- `requirements.txt` → Dependencies
- `.gitignore` → Ignored files

---

## ▶️ How to Run

1. Create virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
Install dependencies:

pip install -r requirements.txt
Run model script (to generate .pkl files):

python movie_recommender.py

Start Streamlit app:

streamlit run app.py

📊 Dataset

TMDB 5000 Movie Dataset (from Kaggle)

