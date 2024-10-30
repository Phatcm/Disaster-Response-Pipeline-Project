# Disaster Response Pipeline Project

This project is designed to assist in the categorization of emergency messages during disaster situations. Using natural language processing (NLP) techniques and machine learning, the model classifies messages into categories such as "medical help," "food," and "water," among others. This application can streamline emergency response by directing messages to the appropriate disaster relief agencies based on the identified categories.

## Project Structure

The project is structured into the following main components:
1. **ETL Pipeline**: Extracts data from a dataset, cleans it, and stores it in a SQLite database.
2. **ML Pipeline**: Builds a machine learning model to classify disaster messages.
3. **Web Application**: A Flask-based web app to allow users to input messages and receive classification results.

## Getting Started

These instructions will guide you through setting up and running the project on your local machine.

### Prerequisites

Make sure you have the following installed:
- Python 3.x
- Necessary Python packages listed in `requirements.txt` (install using `pip install -r requirements.txt`)
- NLTK data packages (`punkt` and `wordnet`)

### Project Files

- **data/process_data.py**: ETL pipeline to clean and store data in a SQLite database.
- **models/train_classifier.py**: ML pipeline to train and save the classifier.
- **app/run.py**: Script to run the Flask web app.

## 1. Running the ETL Pipeline

The ETL pipeline extracts, cleans, and loads data into an SQLite database. To run the ETL pipeline:

1. Navigate to the `data/` directory:
   ```
    cd data
   ```
2. Run the following command, specifying input CSV file paths and the output SQLite database path:
   ```
    python process_data.py disaster_messages.csv disaster_categories.csv Disaster_Response.db
    ```
    Input: <CSV files (disaster_messages.csv and disaster_categories.csv)>.
    Output: <SQLite database path (Disaster_Response.db)>.

## 2. Building the ML Pipeline

The ML pipeline loads data from the SQLite database, trains a classification model, and saves the model to a file.

Navigate to the models/ directory:
```
cd ../models
```
Run the following command to train and save the model:

```
python train_classifier.py ../data/Disaster_Response.db classifier.pkl
```
Input: <SQLite database (Disaster_Response.db).>
Output: <Trained model (classifier.pkl).>

## 3. Running the Web Application
The web app allows you to input a message and view its classification across categories.

Ensure that the trained model (classifier.pkl) and the SQLite database (Disaster_Response.db) are in place.

Navigate to the app/ directory:

```
cd ../app
```
Run the web app:

```
python run.py
```
Open your web browser and go to http://0.0.0.0:3000/ to access the application.

### Using the Web App
- Input a message: Type a message related to a disaster or emergency situation.
- Get predictions: The app will display the message categories predicted by the model.

### Additional Notes
- NLTK data: If you encounter any issues with missing punkt or wordnet data, make sure to download them by running:
```
import nltk
nltk.download('punkt')
nltk.download('wordnet')
```
- Configuration: Update configurations and parameters in config.py as necessary for your environment.

## Acknowledgments
This project is part of the Udacity Data Science Nanodegree Program. Special thanks to the instructors and the community for their guidance and support.
