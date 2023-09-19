
# Reliance Stock Price Prediction

## Project Description

In this project, I aim to predict the stock prices of Reliance Industries using historical data. The primary objective is to employ machine learning models to understand stock price trends and make educated guesses about future prices. This can help both individual investors and financial analysts make informed decisions.

### Goals
1. **Data Preprocessing**: Clean and prepare the historical stock data for analysis.
2. **Exploratory Data Analysis (EDA)**: Analyze the stock prices to identify trends and patterns.
3. **Feature Engineering**: Generate new features that can improve the prediction accuracy of the machine learning model.
4. **Model Building**: Implement machine learning algorithms to predict future stock prices.
5. **Evaluation**: Evaluate the model's performance using metrics like RMSE, MAE, etc.

### Requirements
- Python 3.x
- pandas for data manipulation
- NumPy for numerical operations
- seaborn and Matplotlib for data visualization
- scikit-learn for machine learning algorithms

## Installation

To set up the project environment, follow these steps:

1. Clone the GitHub repository to your local machine.
\```bash
git clone <repository_url>
\```
2. Navigate to the project directory.
\```bash
cd <project_directory>
\```
3. Install the required Python packages.
\```bash
pip install -r requirements.txt
\```

## Data Preprocessing

I started the project by importing essential libraries like NumPy, pandas, seaborn, and Matplotlib. After that, I loaded the historical stock data of Reliance Industries from a CSV file into a pandas DataFrame. I performed basic data cleaning tasks like handling missing values and duplicates to prepare the data for analysis.

\```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("reliance.csv")
df.dropna(inplace=True)
\```

## Exploratory Data Analysis (EDA)

In this section, I performed a thorough analysis of the stock prices to identify trends, patterns, and potential features that could aid in the prediction. I created stock analysis charts to visualize different stock parameters such as Open Price, Close Price, High Price, and Low Price. Additionally, a separate line chart was developed to visualize the closing stock price.

## Feature Engineering

After the EDA, I focused on generating new features from the existing data to improve the model's prediction capabilities. The data was scaled using MinMaxScaler, and a time series dataset was created (details truncated due to code limitations).

\```python
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))
closedf=scaler.fit_transform(np.array(closedf).reshape(-1,1))
\```

## Model Building & Evaluation

I used two machine learning models for the stock price prediction task: Support Vector Regression (SVR) and Random Forest Regressor. The models were trained on the historical data and then evaluated using metrics like RMSE, MAE, and the R2 score.

\```python
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

svr_rbf = SVR(kernel= 'rbf', C= 1e2, gamma= 0.1)
svr_rbf.fit(X_train, y_train)

regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)
regressor.fit(X_train, y_train)
\```

## Usage

To run the project:

1. Ensure that you've followed the installation steps and have all the necessary packages installed.
2. Run the Jupyter Notebook `reliance stock price prediction.ipynb`.

## Technologies Used

- Python
- pandas
- NumPy
- seaborn
- Matplotlib
- scikit-learn

## Credits and License

This project makes use of data from public financial databases and libraries from the Python ecosystem.

