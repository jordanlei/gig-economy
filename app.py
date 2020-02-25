from flask import Flask, request
import numpy as np
import json
from model import generate_calendar_matrix, traffic_data, events_data, weighted_average, edge_link, simple_model

CALENDAR_SIZE = (24, 7)
app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello World!'

@app.route('/get_traffic')
def get_traffic():
    df = generate_calendar_matrix(traffic_data())
    return df.to_json(orient='columns')

@app.route('/get_events')
def get_events():
    df = generate_calendar_matrix(events_data())
    return df.to_json(orient='columns') 

@app.route('/get_recommendations', methods=['POST'])
def get_recommendations():
    if request.method == 'POST':
        user = request.json
        weights = {'traffic': 1, 
                  'events' : 1}

        weights['traffic'] = (-1 * (user['extentFrustratedInTraffic'] - 3))/2
        # mask = np.ones(CALENDAR_SIZE)
        arr = user['availability']
        # print(arr)
        mask = np.asarray(arr)
        mask[:user['earliestHourWillingToWork']] = 0
        mask[user['latestHourWillingToWork']:] = 0
        if not user['drivesOnWeekends']:
            mask[:, 5:] = 0
        if not user['drivesOnWeekdays']:
            mask[:, :5] = 0
        mask = mask.astype(int)
        model_output = simple_model(weights, mask, hours = user['hoursPerWeek'], verbose= False)
        df = generate_calendar_matrix(model_output)
        temp = np.empty(mask.shape, dtype='int')
        dic = {'recommendation' : temp.tolist()}
        return json.dumps(dic)
        return df.to_json(orient='columns') 
    return "Error"


if __name__ == '__main__':
    app.run()