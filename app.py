from flask import Flask, request
import numpy as np
import json
from model import generate_calendar_matrix, traffic_data, events_data, weighted_average, edge_link, simple_model, get_missing_dates



app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello World!!'

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
        postdata = request.json
        user = postdata['user']
        availability = postdata['availability']

        weights = {'traffic': 1, 
                  'events' : 1}
        weights['traffic'] = -1 *(user['extentFrustratedInTraffic'] - 3)/2
        
        dates = []
        available_times = []
        for date in availability:
            dates.append(date)
        dates = dates + get_missing_dates(availability)
        dates = sorted(dates)
        
        available_times = []
        for date in dates:
            if date in availability:
                available_times.append(availability[date])
            else:
                available_times.append([0 for i in range(24)])

        availability_dic = {}
        for idx, date in enumerate(dates):
            availability_dic[date] = available_times[idx]
        return json.dumps(availability_dic, sort_keys=True)

        # CALENDAR_SIZE = (24, len(dates))

        # mask = np.ones(CALENDAR_SIZE)
        # # mask = np.asarray(arr)

        # mask[:user['earliestHourWillingToWork']] = 0
        # mask[user['latestHourWillingToWork']:] = 0

        # #
        # if not user['drivesOnWeekends']:
        #     mask[:, 5:] = 0
        # if not user['drivesOnWeekdays']:
        #     mask[:, :5] = 0
        # mask = mask.astype(int)
        # model_output = simple_model(weights, mask, hours = user['hoursPerWeek'], verbose= False)

        # #simple_model --> json
        # df = generate_calendar_matrix(model_output)
        # temp = np.empty(mask.shape, dtype='int')
        # # dic = {'recommendation' : temp.tolist()}
        # # return json.dumps(dic, sort_keys=True)
        # return df.to_json(orient='columns') 

    return "Error"


if __name__ == '__main__':
    app.run()