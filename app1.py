from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

# Initialize the Flask app
app = Flask(__name__)

# Load the model
model = joblib.load('house_price_prediction_model.pkl')

# Simple login data storage (not suitable for production)
login_data = {'ryuk': '5155', 'pramod': 'kanade'}

# Define routes
@app.route('/')
def index():
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login():
    username = request.form['username']
    password = request.form['password']
    
    if username in login_data and login_data[username] == password:
        return redirect(url_for('predict_page'))
    else:
        return render_template('login.html', message='Invalid username or password.')

@app.route('/predict', methods=['GET', 'POST'])
def predict_page():
    if request.method == 'POST':
        area = request.form['area']
        city = request.form['city']
        sqft = float(request.form['sqft'])
        bhk = int(request.form['bhk'])

        # Make a prediction using the model
        prediction = model.predict([[bhk, sqft]])
        
        # Convert the prediction to INR
        prediction_inr = prediction[0] * 74.5  # Assuming 1 USD = 74.5 INR
        
        # Display the prediction on the result page
        return render_template('predict.html', prediction=prediction_inr)
    
    return render_template('predict.html')

@app.route('/logout')
def logout():
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)