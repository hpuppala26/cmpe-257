# # from flask import Flask, request, render_template
# # from tensorflow.keras.models import load_model
# # import numpy as np
# # import pandas as pd
# # from sklearn.preprocessing import MinMaxScaler
# # import joblib

# # app = Flask(__name__)
# # port = 5100
# # # Load the pre-trained model
# # model = load_model('bike_sharing_model.h5')

# # # Assuming the scaler was saved previously after fitting
# # scaler = joblib.load('scaler.pkl')

# # @app.route('/', methods=['GET', 'POST'])
# # def index():
# #     if request.method == 'POST':
# #         # Get data from POST request
# #         time_slot = request.form.get('time_slot')
# #         isWeekday = request.form.get('isWeekday') == 'True'
# #         isPeakHour = request.form.get('isPeakHour') == 'True'
        
# #         # Make prediction
# #         predicted_trip_count = predict_trip_count(time_slot, isWeekday, isPeakHour)
        
# #         # Render the result in the HTML template
# #         return render_template('index.html', trip_count=predicted_trip_count)
    
# #     # If not a POST request, just render the form
# #     return render_template('index.html', trip_count=None)

# # def predict_trip_count(time_slot, isWeekday, isPeakHour):
# #     input_data = pd.DataFrame([[time_slot, isWeekday, isPeakHour]], 
# #                               columns=['time_slot', 'isWeekday', 'isPeakHour'])
# #     input_scaled = scaler.transform(input_data)
# #     input_reshaped = input_scaled.reshape((1, 1, input_scaled.shape[1]))
# #     predicted_count = model.predict(input_reshaped)
# #     return predicted_count[0][0]

# # if __name__ == '__main__':
# #     app.run(debug=True)





# from flask import Flask, request, render_template
# from tensorflow.keras.models import load_model
# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import MinMaxScaler
# import joblib
# from datetime import datetime

# app = Flask(__name__)
# port = 5100
# # Load the pre-trained model
# model = load_model('bike_sharing_model.h5')
# scaler = joblib.load('scaler.pkl')

# @app.route('/', methods=['GET', 'POST'])
# def index():
#     if request.method == 'POST':
#         date_input = request.form.get('date_input')
#         time_input = request.form.get('time_input')
#         station = request.form.get('station')

#         time_slot, isWeekday, isPeakHour = process_date_time(date_input, time_input)

#         predicted_trip_count = predict_trip_count(time_slot, isWeekday, isPeakHour, station)
        
#         return render_template('index.html', trip_count=predicted_trip_count, stations=get_stations())
    
#     return render_template('index.html', trip_count=None, stations=get_stations())

# def process_date_time(date_str, time_str):
#     date_time = datetime.strptime(f"{date_str} {time_str}", '%Y-%m-%d %H:%M')
#     hour = date_time.hour
#     time_slot = (hour // 4) % 6
#     isWeekday = date_time.weekday() < 5
#     isPeakHour = hour in range(6, 10) or hour in range(14, 18)
#     return time_slot, isWeekday, isPeakHour

# def predict_trip_count(time_slot, isWeekday, isPeakHour, station):
#     # Add station handling logic if needed
#     input_data = pd.DataFrame([[time_slot, isWeekday, isPeakHour]], 
#                               columns=['time_slot', 'isWeekday', 'isPeakHour'])
#     input_scaled = scaler.transform(input_data)
#     input_reshaped = input_scaled.reshape((1, 1, input_scaled.shape[1]))
#     predicted_count = model.predict(input_reshaped)
#     return predicted_count[0][0]

# def get_stations():
#     # Replace this with actual station retrieval logic if needed
#     return ['Station1', 'Station2', 'Station3', 'Station4', 'Station5', 'Station6', 'Station7']

# if __name__ == '__main__':
#     app.run(debug=True)





from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
from datetime import datetime

app = Flask(__name__)
# Load the pre-trained model
model = load_model('bike_sharing_model.h5')
scaler = joblib.load('scaler.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        date_input = request.form.get('date_input')
        time_input = request.form.get('time_input')
        station = request.form.get('station')

        time_slot, isWeekday, isPeakHour = process_date_time(date_input, time_input)
        chance_percentage, emoji, message = predict_trip_count(time_slot, isWeekday, isPeakHour, station)
        
        # Add your logic here to determine the correct gif_url based on the emoji
        gif_url = f'https://t4.ftcdn.net/jpg/01/15/20/75/360_F_115207580_US2etunH78I7iMYHOoNVvxQTCIdoPdRj.jpg'  # Placeholder for actual gif paths

        return render_template('index.html', gif_url=gif_url, emoji=emoji, message=message, stations=get_stations())
    
    return render_template('index.html', trip_count=None, stations=get_stations())


@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/support', methods=['GET', 'POST'])
def support():
    if request.method == 'POST':
        return render_template('thank_you.html')
    return render_template('support.html')

def process_date_time(date_str, time_str):
    date_time = datetime.strptime(f"{date_str} {time_str}", '%Y-%m-%d %H:%M')
    hour = date_time.hour
    time_slot = (hour // 4) % 6
    isWeekday = date_time.weekday() < 5
    isPeakHour = hour in range(6, 10) or hour in range(14, 18)
    return time_slot, isWeekday, isPeakHour

def predict_trip_count(time_slot, isWeekday, isPeakHour, station):
    # Dummy station handling
    station_mapping = {'Santa Clara St at 7th St': 0, '1st St at San Carlos St': 1, 'Saint James Park': 2, 'Santa Clara St at Almaden Blvd': 3, 'Julian St at 6th St':4, '9th St at San Fernando St':5, 'Fountain Alley at S 2nd St':6, '19th St at William St': 7, '22nd St at William St':8}
    station_index = station_mapping.get(station, 0)
    
    input_data = pd.DataFrame([[time_slot, isWeekday, isPeakHour]], 
                              columns=['time_slot', 'isWeekday', 'isPeakHour'])
    input_scaled = scaler.transform(input_data)
    input_reshaped = input_scaled.reshape((1, 1, input_scaled.shape[1]))
    predicted_count = model.predict(input_reshaped)[0][0]
    
    # Logic to calculate the message and emoji based on predicted_count
    if predicted_count > 3:
        chance_percentage = np.random.uniform(90, 95)
        emoji = 'smiley'
        message = 'Congratulations! The chances of you getting a bike at your selected station is going to be {:.0f}%.'
    elif predicted_count > 2:
        chance_percentage = np.random.uniform(70, 80)
        emoji = 'doubtful'
        message = 'Hey! The chances of you getting a bike at your selected station is going to be {:.0f}%. But it is quite possible that you will get the bike.'
    else:
        chance_percentage = np.random.uniform(50, 60)
        emoji = 'sad'
        message = 'Hey, Sorry! The chances of you getting a bike at your selected station is going to be {:.0f}%. But, you can always try your luck!'
    
    return chance_percentage, emoji, message.format(chance_percentage)

def get_stations():
    # Replace with actual logic to retrieve station names
    return ['None','Santa Clara St at 7th St', '1st St at San Carlos St', 'Saint James Park', 'Santa Clara St at Almaden Blvd', 'Julian St at 6th St', '9th St at San Fernando St', 'Fountain Alley at S 2nd St', '19th St at William St', '22nd St at William St']

if __name__ == '__main__':
    app.run(debug=True)
