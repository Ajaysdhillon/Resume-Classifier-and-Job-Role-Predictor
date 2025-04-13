🚀 Automated Resume Classification and Job Role Prediction
Welcome to the Automated Resume Classification and Job Role Prediction project! This project utilizes Natural Language Processing (NLP) and Machine Learning to predict job roles based on resume content. It showcases techniques such as text preprocessing, feature extraction using TF-IDF, and model training with Naive Bayes and SVM classifiers.

✨ Project Overview
In this project, we automate the process of classifying resumes into various job roles. Here's what the project entails:

Text Preprocessing: Cleaning and normalizing resume text, including removing unnecessary characters, digits, stopwords, and performing tokenization and lemmatization.

Feature Extraction: Extracting meaningful features from resumes using TF-IDF (Term Frequency-Inverse Document Frequency) vectorization.

Model Training: Training Naive Bayes and SVM classifiers to predict the job roles of candidates.

Model Evaluation: Evaluating the models based on various classification metrics like accuracy, precision, recall, and F1-score.

Model Persistence: Saving the trained models and vectorizers for future use using joblib.

🔧 Prerequisites
Before you run the project, make sure you have the following installed:

Python 3.x

Libraries:

pandas

nltk

scikit-learn

joblib

📊 Dataset
The dataset consists of resumes in CSV format with the following expected columns:

resume_text: The raw content of the resume (in text format).

profession: The predicted job role (the label).
🛠 Project Workflow
1. Data Preprocessing
We clean the resume text by:

Lowercasing the text.

Removing special characters and digits.

Tokenizing the text.

Removing stopwords.

Lemmatizing the tokens.

2. Feature Extraction
We use TF-IDF Vectorization to transform the cleaned text into numerical features, which can be understood by machine learning models.

3. Model Training
We train two machine learning models:

Naive Bayes: A fast and effective text classification algorithm.

SVM (Support Vector Machine): A more robust classifier optimized using GridSearchCV.

4. Model Evaluation
We evaluate the models based on:

Accuracy: How often the classifier makes the correct prediction.

Precision: How many of the predicted positives are actually positive.

Recall: How many of the actual positives are correctly identified.

F1-score: The harmonic mean of precision and recall.

5. Model Saving
Finally, we save the trained models and vectorizer using joblib for future predictions.

