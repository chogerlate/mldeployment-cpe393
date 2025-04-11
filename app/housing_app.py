from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the trained model and model info
with open("housing_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("housing_model_info.pkl", "rb") as f:
    model_info = pickle.load(f)

@app.route("/")
def home():
    return "Housing Price Prediction Model API is Running"

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        
        # Check if features key exists
        if "features" not in data:
            return jsonify({"error": "Missing 'features' key in request"}), 400
        
        # Handle both single house and multiple houses
        inputs = data["features"]
        
        # If single house, convert to list of houses
        if not isinstance(inputs, list):
            return jsonify({"error": "Features must be a list of houses"}), 400
            
        # Check if it's a list of dictionaries (houses)
        if not inputs:
            return jsonify({"error": "Empty features list"}), 400
            
        # Validate input structure
        all_columns = model_info['all_columns']
        
        # Convert input data to DataFrame with proper column order
        try:
            if isinstance(inputs[0], dict):
                # List of dictionaries
                df = pd.DataFrame(inputs)
            else:
                # Assume list of lists with values in the same order as training data
                df = pd.DataFrame(inputs, columns=all_columns)
                
            # Ensure all required columns are present
            for col in all_columns:
                if col not in df.columns:
                    return jsonify({"error": f"Missing column '{col}' in input"}), 400
            
            # Ensure only expected columns are present
            extra_cols = [col for col in df.columns if col not in all_columns]
            if extra_cols:
                return jsonify({"error": f"Unexpected columns in input: {extra_cols}"}), 400
                
            # Reorder columns to match the training order
            df = df[all_columns]
            
        except Exception as e:
            return jsonify({"error": f"Invalid input format: {str(e)}"}), 400
        
        # Make predictions
        predictions = model.predict(df).tolist()
        
        # Return single prediction or list of predictions
        if len(predictions) == 1:
            return jsonify({"prediction": predictions[0]})
        else:
            return jsonify({"predictions": predictions})
            
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/model_info", methods=["GET"])
def get_model_info():
    return jsonify({
        "model_type": "Random Forest Regressor",
        "target": "housing price",
        "features": model_info['all_columns'],
        "categorical_features": model_info['categorical_cols'],
        "numerical_features": model_info['numerical_cols']
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9000) 