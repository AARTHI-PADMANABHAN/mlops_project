import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv('../data/sampregdata.csv')

# Select the best X based on correlation (assume 'x1' is the best predictor)
X = df[['x1']]
y = df['y']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Save model
with open('../models/model_v1.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model version 1 trained and saved.")
