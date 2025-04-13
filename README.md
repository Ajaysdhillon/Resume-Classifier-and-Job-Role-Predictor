Automated Resume Classification and Job Role Prediction
This project automates the classification of resumes into different job roles using Natural Language Processing (NLP) and machine learning models. The goal is to predict a candidate's job role based on the content of their resume. The project includes data preprocessing, feature extraction, model training, and evaluation.

Project Overview
This project demonstrates how to classify resumes into job roles using machine learning. It involves preprocessing raw resume data, extracting meaningful features from the text, and applying classification models to predict the job roles. The project uses the following steps:

Text Preprocessing: Clean and normalize resume text by removing unnecessary characters, digits, and stopwords, followed by tokenization and lemmatization.

Feature Extraction: Transform the cleaned text into numerical features using TF-IDF (Term Frequency-Inverse Document Frequency).

Model Training: Train two machine learning models—Naive Bayes and Support Vector Machine (SVM)—to classify the resumes.

Hyperparameter Tuning: Perform GridSearchCV to optimize the parameters for the SVM model.

Model Evaluation: Evaluate models based on accuracy and classification metrics.

Model Saving: Save the trained models and TF-IDF vectorizers using joblib for future use.

Features
Text Preprocessing: The text data undergoes cleaning, tokenization, stopword removal, and lemmatization.

TF-IDF Vectorization: Transforms text data into numerical vectors using the TF-IDF method.

Machine Learning Models: Trains Naive Bayes and SVM models for classification.

GridSearchCV: Performs hyperparameter tuning for the SVM model to select the best parameters.

Model Persistence: Saves the trained models and vectorizers to disk for later use.

Prerequisites
Python 3.x

Required libraries:

pandas

nltk

scikit-learn

joblib
Dataset
The dataset consists of resumes in text format, with the following expected columns:

resume_text: The raw text content of the resume.

profession: The job role/label associated with the resume.

Please ensure your dataset is in a CSV file, and update the code to point to the correct dataset path.

Project Workflow
Data Preprocessing:

Converts all text to lowercase.

Removes special characters and digits.

Tokenizes the text and removes stopwords.

Lemmatizes the tokens for consistency.

Feature Extraction:

Converts the cleaned text into numerical features using the TF-IDF vectorizer.

Model Training:

Trains two machine learning models:

Naive Bayes: A probabilistic classifier that is fast and effective for text classification.

SVM: A robust model for text classification, with hyperparameter tuning using GridSearchCV.

Model Evaluation:

The models are evaluated based on accuracy, precision, recall, and F1-score.

The classification report is printed for both models.

Model Saving:

The best SVM model and the TF-IDF vectorizer are saved using joblib for future use.

