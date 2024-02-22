# Sentiment Analysis Web Application

This Sentiment Analysis Web Application leverages Python and Natural Language Processing (NLP) techniques to analyze the sentiment of tweets. Using a trained model, the application can classify tweets as positive, negative, or neutral. This project includes a Streamlit web application for an interactive user experience.

## Project Structure

```
sentiment-analysis-webapp/
│
├── app/                        # Directory for the Streamlit webapp
│   ├── __init__.py             # Makes app a Python module
│   └── streamlit_app.py        # Streamlit app main script
│
├── data/                       # Data directory for storing datasets
│   ├── train.csv               # Training dataset
│   └── test.csv                # Testing dataset
│
├── models/                     # Directory for saved model files
│   ├── sentiment_model.pkl     # Serialized model to be used by the app
│   └── tfidf_vectorizer.pkl    # Tfidf_vectorizer model to be used by the app
│
├── notebooks/                  # Jupyter notebooks for exploration and presentations
│   └── model_development.ipynb # Notebook for model training and evaluation
│
├── src/                        # Source code for the project
│   ├── __init__.py             # Makes src a Python module
│   ├── data_preprocessing.py   # Script for data cleaning and preprocessing
│   ├── feature_extraction.py   # Script for converting text data into vectors
│   └── train_model.py          # Script for model training and evaluation
│
├── tests/                      # Unit tests and integration tests
│   └── test_preprocessing.py   # Test cases for data preprocessing
│
├── .gitignore                  # Specifies intentionally untracked files to ignore
├── README.md                   # Project overview, setup, and usage instructions
└── requirements.txt            # List of dependencies to be installed
```

## Setup and Installation

1. **Clone the repository**

   ```sh
   git clone https://github.com/saimaharana4/sentiment-analysis-webapp.git
   cd sentiment-analysis-webapp
   ```

2. **Create and activate a virtual environment**

   - For Windows:

     ```sh
     python -m venv venv
     venv\Scripts\activate
     ```

   - For macOS/Linux:

     ```sh
     python3 -m venv venv
     source venv/bin/activate
     ```

3. **Install dependencies**

   ```sh
   pip install -r requirements.txt
   ```

4. **Train the model**

   Navigate to the project root directory and run:

   ```sh
   python src/train_model.py
   ```

   This will train the sentiment analysis model and save it, along with the TF-IDF vectorizer, in the `models/` directory.

5. **Run the Streamlit Web Application**

   ```sh
   SET PYTHONPATH=%PYTHONPATH%;D:\sentiment-analysis-webapp
   streamlit run app/streamlit_app.py
   ```

   This command will start the Streamlit server and open the web application in your default browser.

## Usage

Once the Streamlit web application is running, we can analyze the sentiment of tweets by entering the text of the tweet into the input field and clicking the "Predict Sentiment" button. The application will display the sentiment classification based on the input text.


---


