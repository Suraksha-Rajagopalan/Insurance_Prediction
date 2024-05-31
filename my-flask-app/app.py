from flask import Flask, request, render_template
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load pre-trained models
random_forest = joblib.load('models/random_forest.joblib')
gradient_boosting = joblib.load('models/gradient_boosting.joblib')
linear_regression = joblib.load('models/linear_regression.joblib')
svr = joblib.load('models/svr.joblib')

# Initialize StandardScaler
scaler = StandardScaler()
scaler.mean_ = joblib.load('models/scaler_mean_.joblib')
scaler.scale_ = joblib.load('models/scaler_scale_.joblib')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract feature values from form
    features = [float(x) for x in request.form.values()]
    features = np.array(features).reshape(1, -1)
    features_scaled = scaler.transform(features)
    
    # Make predictions with all models
    predictions = {
        'Random Forest': float(random_forest.predict(features_scaled)[0]),
        'Gradient Boosting': float(gradient_boosting.predict(features_scaled)[0]),
        'Linear Regression': float(linear_regression.predict(features_scaled)[0]),
        'Support Vector Regression': float(svr.predict(features_scaled)[0]),
    }

    # Ensure all predictions are numerical values
    for model, prediction in predictions.items():
        if not isinstance(prediction, (int, float)):
            predictions[model] = float(prediction)

    # Format predictions with two digits after the decimal point
    formatted_predictions = {model: '{:.2f}'.format(prediction) for model, prediction in predictions.items()}

    # Render the template with formatted predictions
    return render_template('index.html', predictions=formatted_predictions)

@app.route('/plots')
def plots():
    return render_template('plots.html')

if __name__ == '__main__':
    app.run(debug=True)
