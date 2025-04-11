import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load the housing dataset
df = pd.read_csv("Housing.csv")

# Split features and target
X = df.drop('price', axis=1)
y = df['price']

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

# Create the model pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Train the model
model.fit(X, y)

# Save the model for the housing price prediction API
with open("app/housing_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save column information for the API
model_info = {
    'categorical_cols': categorical_cols,
    'numerical_cols': numerical_cols,
    'all_columns': X.columns.tolist()
}

with open("app/housing_model_info.pkl", "wb") as f:
    pickle.dump(model_info, f)

print("Housing price prediction model trained and saved successfully.") 