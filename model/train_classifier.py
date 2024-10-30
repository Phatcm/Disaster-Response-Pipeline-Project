# Import required libraries
import sys
import numpy as np
import pandas as pd
import re
import pickle
from sqlalchemy import create_engine

import nltk
nltk.download(['punkt', 'wordnet', 'stopwords', 'averaged_perceptron_tagger'])
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from tqdm import tqdm
from sklearn.base import clone


def load_data(database_filepath):
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('DisasterResponse', engine)

    X = df['message']
    Y = df.iloc[:, 4:]
    category_names = Y.columns.tolist()

    # Debugging prints
    print(f"Loaded data: {df.head()}")
    print(f"Features shape: {X.shape}, Labels shape: {Y.shape}, Category names: {category_names}")

    return X, Y, category_names


def tokenize(text):
    """
    Process text by tokenizing, lemmatizing, and removing stopwords.

    Parameters:
        text (str): Input text for processing.

    Returns:
        list: Processed list of tokens.
    """
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r"[^a-zA-Z0-9]", " ", text)
        tokens = text.split()
        tokens = [word for word in tokens if word not in stopwords.words("english")]
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
        return tokens
    return []


def build_model(classifier=RandomForestClassifier()):
    """
    Set up a machine learning pipeline and configure GridSearch for hyperparameter tuning.

    Parameters:
        classifier (estimator, optional): Estimator for classification, default is RandomForestClassifier.

    Returns:
        GridSearchCV: Configured pipeline with specified parameters and verbosity.
    """
    pipeline = Pipeline([
        ('vectorizer', CountVectorizer(tokenizer=tokenize)),
        ('tfidf_transformer', TfidfTransformer()),
        ('multi_output_classifier', MultiOutputClassifier(classifier))
    ])

    parameters = {
        "multi_output_classifier__estimator__n_estimators": [50, 100],
        "multi_output_classifier__estimator__min_samples_split": [2, 3]
    }

    model = GridSearchCV(pipeline, param_grid=parameters, verbose=3)  # Set verbosity for progress feedback
    return model

def train_model_with_progress(model, X_train, Y_train):
    """
    Train the model with a simple progress indication.

    Parameters:
        model: Configured GridSearchCV model.
        X_train: Training data.
        Y_train: Training labels.

    Returns:
        Fitted model.
    """
    try:
        print("Fitting the model...")
        model.fit(X_train, Y_train)  # Fit the model
        print("Model training completed.")
        return model  # Return the fitted model
    except Exception as e:
        print(f"Error in train_model_with_progress: {e}")
        return None  # Return None if an error occurs


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate model predictions using classification metrics for each category.

    Parameters:
        model: Trained model for generating predictions.
        X_test: Test messages.
        Y_test: True category values.
        category_names: List of category names.

    Returns:
        None
    """
    predictions = model.predict(X_test)

    for i, category in enumerate(category_names):
        print(f"Category: {category}\n")
        print(classification_report(Y_test[category], predictions[:, i]))


def save_model(model, model_filepath):
    """
    Save the trained model to a file.

    Parameters:
        model: Trained machine learning model.
        model_filepath (str): Path to save the model as a pickle file.

    Returns:
        None
    """
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)

def main():
    try:
        # Initialize variables
        database_filepath = None
        model_filepath = None
        
        # Check command line arguments
        if len(sys.argv) == 3:
            database_filepath, model_filepath = sys.argv[1:3]  # Correcting index to 1:3
            print(f"Loading data from: {database_filepath}")
        else:
            # Prompt user for database file and model file paths
            database_filepath = input("Enter the path to the SQLite database (e.g., data/DisasterResponse.db): ")
            model_filepath = input("Enter the filename to save the trained model (e.g., model/classifier.pkl): ")

        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        print(f"Data loaded successfully. X shape: {X.shape}, Y shape: {Y.shape}")
        print(f"Category names: {category_names}\nData split into training and test sets.")

        print("Building model pipeline...")
        model = build_model()

        print("Training model, please wait...")
        trained_model = train_model_with_progress(model, X_train, Y_train)
        if trained_model is None:
            print("Model training failed. Exiting.")
            return

        print("\nEvaluating model performance...")
        evaluate_model(trained_model, X_test, Y_test, category_names)

        print(f"\nSaving trained model to: {model_filepath}")
        save_model(trained_model, model_filepath)
        print("Model saved successfully!")

    except Exception as e:
        print(f"There is an error when accessing the file: {e}")
        
if __name__ == '__main__':
    main()