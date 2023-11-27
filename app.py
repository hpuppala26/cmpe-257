from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib

app = Flask(__name__)
port = 5100
# Load the pre-trained model
model = load_model('bike_sharing_model.h5')

# Assuming the scaler was saved previously after fitting
scaler = joblib.load('scaler.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get data from POST request
        time_slot = request.form.get('time_slot')
        isWeekday = request.form.get('isWeekday') == 'True'
        isPeakHour = request.form.get('isPeakHour') == 'True'
        
        # Make prediction
        predicted_trip_count = predict_trip_count(time_slot, isWeekday, isPeakHour)
        
        # Render the result in the HTML template
        return render_template('index.html', trip_count=predicted_trip_count)
    
    # If not a POST request, just render the form
    return render_template('index.html', trip_count=None)

def predict_trip_count(time_slot, isWeekday, isPeakHour):
    input_data = pd.DataFrame([[time_slot, isWeekday, isPeakHour]], 
                              columns=['time_slot', 'isWeekday', 'isPeakHour'])
    input_scaled = scaler.transform(input_data)
    input_reshaped = input_scaled.reshape((1, 1, input_scaled.shape[1]))
    predicted_count = model.predict(input_reshaped)
    return predicted_count[0][0]

if __name__ == '__main__':
    app.run(debug=True)
