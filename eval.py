from flask import Flask, request
import numpy as np
import json
from model import generate_calendar_matrix, traffic_data, events_data, weighted_average, edge_link, simple_model, get_missing_dates


data = json.load(open('traffic_lovers.json', 'r'))
final_json = []
heatmap = np.zeros((24, 7)) #TALLEY
for postdata in data:
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

    CALENDAR_SIZE = (24, len(dates))
    for day in available_times:
        for idx, time in enumerate(day):
            if idx < user['earliestHourWillingToWork']: day[idx] = 0 
            if idx > user['latestHourWillingToWork']: day[idx] = 0 

    mask = 1 - np.asarray(available_times)
    mask = np.transpose(mask)
    # return json.dumps(available_times) #FOR DEBUGGING available_times
    # return json.dumps(mask.tolist()) #FOR DEBUGGING mask
    # return json.dumps( [mask.shape] ) #FOR DEBUGGING

    model_output = simple_model(weights, mask.astype(bool), user['hoursPerWeek'], CALENDAR_SIZE, dates, verbose= False)
    # return json.dumps((np.transpose(model_output).astype(int)).tolist()) #FOR DEBUGGING. generate_calendear_matrix transposes this matrix to get the true recommendation

    #simple_model --> tranpose (24,x) to (x, 24) --> json
    df = generate_calendar_matrix(model_output, dates, CALENDAR_SIZE)
    df_dict = df.to_dict('dict')

    final_dic = {}
    one_user = []
    for key, val in df_dict.items():
        rec = []
        for k, v in val.items():
            rec.append(1) if v else rec.append(0)
        final_dic[key] = rec
        one_user.append(rec)

    # print(one_user)
    heatmap = heatmap + np.transpose(np.array(one_user))


    # final_json.append(final_dic)

# print(json.dumps(final_json))
print(heatmap)