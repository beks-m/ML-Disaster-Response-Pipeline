# Disaster Response Pipeline Project

### Description
Disaster Response Pipeline Project from Udacity's Data Science Nanodegree.
We will be creating a machine learning pipeline to categorize disaster events so that we can send the messages to an appropriate disaster relief agency. We will train on data set containing real messages that were sent during disaster events from Figure Eight. 

Project consists from three parts:

1. ETL Pipeline
Data cleaning pipeline contained in data/process_data.py:
- Loads the messages and categories datasets
- Merges the two datasets
- Cleans the data
- Stores it in a SQLite database

2. ML Pipeline
Machine learning pipeline contained in model/train_classifier.py:
- Loads data from the SQLite database
- Splits the dataset into training and test sets
- Builds a text processing and machine learning pipeline
- Trains and tunes a model using GridSearchCV
- Outputs results on the test set
- Exports the final model as a pickle file

3. Flask Web App
Web app where an emergency worker can input a new message and get classification results in several categories. The web app also displays visualizations of the data.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Code structure

- app
  - template
    - master.html  # main page of web app
    - go.html  # classification result page of web app
  - run.py  # Flask file that runs app

- data
  - disaster_categories.csv  # data to process 
  - disaster_messages.csv  # data to process
  - process_data.py
  - InsertDatabaseName.db   # database to save clean data to

- models
  - train_classifier.py
  - classifier.pkl  # saved model 

- README.md

