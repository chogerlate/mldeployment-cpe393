from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/")
def home():
    return "ML Model is Running"

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
            
        features = data["features"]
        
        # Check if the input is a single sample or multiple samples
        if not isinstance(features, list):
            return jsonify({"error": "Features must be a list"}), 400
            
        if not features:
            return jsonify({"error": "Empty features list"}), 400
            
        # Check if single sample or multiple samples
        if not isinstance(features[0], list):
            features = [features]  # Convert to list of lists if single sample
            
        # Validate each feature vector has exactly 4 float values
        for i, feature_vector in enumerate(features):
            if len(feature_vector) != 4:
                return jsonify({"error": f"Feature vector at index {i} must contain exactly 4 values"}), 400
                
            # Check if all values are numeric
            if not all(isinstance(val, (int, float)) for val in feature_vector):
                return jsonify({"error": f"All values in feature vector at index {i} must be numeric"}), 400
        
        input_features = np.array(features)
        predictions = model.predict(input_features).tolist()
        confidences = model.predict_proba(input_features).max(axis=1).tolist()
        
        # If it's a single prediction, include confidence score
        if len(predictions) == 1:
            return jsonify({"prediction": predictions[0], "confidence": confidences[0]})
        
        # For multiple predictions, return the list with confidences
        return jsonify({
            "predictions": predictions,
            "confidences": confidences
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9000) #check your port number ( if it is in use, change the port number)
