This is a Sentimental  Analysis project using Python and Natural Language Processing techniques.

Activate the virtual env: venv\Scripts\activate

Here is the project structure.

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
│   └── sentiment_model.pkl     # Serialized model to be used by the app
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
├── LICENSE                     # License for the project
├── README.md                   # Project overview, setup, and usage instructions
└── requirements.txt            # List of dependencies to be installed
