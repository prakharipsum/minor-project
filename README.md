# 📰 Fake News Detection Using Machine Learning

A machine learning-based web application that classifies news articles as **Real** or **Fake** using Natural Language Processing (NLP) techniques and supervised learning algorithms. The system preprocesses textual data, transforms it using TF-IDF vectorization, and predicts news authenticity through a trained classification model. A user-friendly Flask interface enables real-time news verification. Based on the project report "Fake News Detection Using Machine Learning & Deep Learning."

## 🚀 Features

* Real-time Fake News Detection
* Interactive Flask-based Web Interface
* Text Preprocessing and Cleaning
* TF-IDF Feature Extraction
* Machine Learning Classification Model
* Confidence Score Visualization
* Responsive Dashboard with Charts
* Fast and Lightweight Deployment

## 🛠️ Tech Stack

**Programming Language**

* Python

**Machine Learning & NLP**

* Scikit-learn
* TF-IDF Vectorization
* Logistic Regression
* NLTK

**Web Development**

* Flask
* HTML
* CSS
* JavaScript
* Chart.js

**Libraries**

* NumPy
* Pandas
* Joblib

## 📊 Project Workflow

1. Data Collection
2. Data Cleaning & Preprocessing
3. Text Normalization
4. Stopword Removal
5. Stemming
6. TF-IDF Vectorization
7. Model Training
8. Model Evaluation
9. Flask Application Integration
10. Real-Time Prediction

## 📁 Project Structure

```bash
Fake-News-Detection/
│
├── app.py
├── fake_news_model.pkl
├── tfidf_vectorizer.pkl
├── requirements.txt
├── dataset/
├── notebooks/
├── static/
├── templates/
└── README.md
```

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/your-username/fake-news-detection.git
cd fake-news-detection
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

Windows:

```bash
venv\Scripts\activate
```

Linux/Mac:

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Application

```bash
python app.py
```

Open:

```bash
http://127.0.0.1:5000
```

## 🎯 Model Performance

The project utilizes machine learning techniques combined with NLP preprocessing for fake news classification. According to the project report, the Logistic Regression-based model achieved approximately **97.9% accuracy** on the evaluated dataset.

## 📸 Application Features

* Paste any news article
* Click Analyze
* View prediction result
* Check confidence score
* Visualize probability distribution through charts

## 🔮 Future Enhancements

* Deep Learning Models (LSTM, BERT, CNN)
* News Source Credibility Analysis
* Multi-language Support
* API Deployment
* Real-time News Verification Extension
* Explainable AI Predictions

## 👨‍💻 Contributors

* Prakhar Kumar
* Aryan Singh
* Shreya Kasana
* Chirag Choudhary

## 📄 License

This project is developed for educational and research purposes.
