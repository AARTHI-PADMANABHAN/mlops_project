import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv('../data/sampregdata.csv')

# Use two predictors: x1 and x2
X = df[['x1', 'x2']]
y = df['y']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the new model as version 2
with open('../models/model_v2.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model version 2 trained and saved.")
