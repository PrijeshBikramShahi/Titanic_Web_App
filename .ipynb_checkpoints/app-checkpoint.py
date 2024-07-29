from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the model
model = joblib.load('models/titanic_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Retrieve and convert form values
        features = [float(request.form['age']),
                    float(request.form['fare']),
                    float(request.form['sibsp']),
                    float(request.form['parch'])]
        
        final_features = np.array([features])
        
        # Make prediction
        prediction = model.predict(final_features)
        
        return render_template('index.html', prediction_text=f'Prediction: {"Survived" if prediction[0] == 1 else "Did not survive"}')
    except Exception as e:
        return render_template('index.html', prediction_text=f'An error occurred: {e}')

if __name__ == "__main__":
    app.run(debug=True, port=5001)
