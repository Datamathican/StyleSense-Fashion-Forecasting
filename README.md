```
# StyleSense: Fashion Forward Forecasting

## Project Overview
This project builds a predictive machine learning pipeline for "StyleSense," a rapidly growing online women's clothing retailer. As customer growth has created a backlog of product reviews with missing recommendation data, this pipeline automates the prediction of whether a customer would recommend a product based on the text of their review and other demographic and product features. 

By predicting recommendations, StyleSense can gain insights into customer satisfaction, identify trending products, and improve the shopping experience.

## Project Environment & Dependencies
This project was developed using a Jupyter Notebook environment. The machine learning pipeline utilizes `scikit-learn` along with `spaCy` for Natural Language Processing (NLP).

**Requirements:**
- `python >= 3.6`
- `pandas`
- `numpy`
- `scikit-learn`
- `spacy`

To install the required packages, run:
`pip install -r requirements.txt`

You will also need to download the English NLP model for spaCy:
`python -m spacy download en_core_web_sm`

## Repository Files
- `notebook.ipynb`: The primary Jupyter Notebook containing the full machine learning pipeline. It includes data exploration, preprocessing for numerical/categorical/text data, model training using a Random Forest Classifier, hyperparameter tuning via GridSearchCV, and final model evaluation.
- `requirements.txt`: Contains the list of Python packages required to run the notebook successfully.
- `README.md`: This file, providing an overview of the project and repository contents.

## Pipeline Architecture
The machine learning pipeline is designed to handle mixed data types without data leakage, integrating preprocessing and prediction into a single structure:
1. **Numerical Features:** Imputed using the median and scaled using `StandardScaler` (e.g., Age, Positive Feedback Count).
2. **Categorical Features:** Imputed with a constant value and transformed using `OneHotEncoder` (e.g., Division Name, Department Name).
3. **Text Features:** Normalized, lemmatized, and stripped of stop words using a custom `spaCy` tokenizer, then vectorized using `TfidfVectorizer` (e.g., Review Text).
4. **Classifier:** A `RandomForestClassifier` serves as the final prediction model to output the binary recommendation indicator.
```
