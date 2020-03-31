import numpy as np
import pandas as pd
from datetime import date, timedelta, datetime
import calendar


#BOSTON TRAFFIC WEEKDAY for 2019 https://www.tomtom.com/en_gb/traffic-index/boston-traffic
traffic_boston = {"Monday" : [4, 3, 3, 0, 0, 7, 25, 40, 42, 25, 17, 16, 18, 18, 26, 34, 40, 47, 31, 15, 11, 9, 7, 5],
"Tuesday" : [4, 3, 2, 0, 0, 9, 29, 49, 53, 34, 22, 18, 20, 20, 30, 40, 46, 54, 36, 18, 12, 10, 8, 5],
"Wednesday" : [3, 2, 2, 0, 0, 8, 28, 47, 50, 32, 21, 19, 20, 20, 30, 41, 48, 57, 39, 20, 13, 11, 9, 6],
"Thursday" : [5, 4, 3, 1, 0, 7, 27, 46, 49, 32, 21, 19, 21, 20, 31, 43, 52, 60, 42, 21, 14, 11, 10, 7],
"Friday" : [4, 3, 1, 0, 0, 4, 17, 30, 31, 21, 19, 21, 25, 27, 40, 51, 53, 51, 34, 19, 14, 12, 11, 8],
"Saturday" : [5, 4, 3, 0, 0, 0, 0, 1, 5, 10, 15, 20, 24, 24, 23, 22, 20, 20, 17, 14, 11, 10, 9, 7],
"Sunday" : [5, 3, 3, 0, 0, 0, 0, 0, 1, 4, 9, 14, 18, 18, 17, 16, 16, 16, 14, 11, 10, 8, 6, 5]}



def findDay(date): 
    '''
    Input date example: '2020-02-26'
    Output: 'Wednesday'
    '''
    born = datetime.strptime(date, '%Y-%m-%d').weekday() 
    return (calendar.day_name[born]) 

def get_missing_dates(availability):
    '''
    Take in JSON-format avalability and output missing date in list in string format
    '''
    dates = []
    for date in availability:
        d = datetime.strptime(date, '%Y-%m-%d')
        dates.append(d)
        
    date_set = set(dates[0] + timedelta(x) for x in range((dates[-1] - dates[0]).days))
    missing = sorted(date_set - set(dates))
    missing_dates = []
    for date in missing:
        missing_dates.append(date.strftime("%Y-%m-%d"))

    return missing_dates

def generate_calendar_matrix(arr, dates, CALENDAR_SIZE): 
    '''
    Generates a calendar matrix based off
    an optional numpy input array. If no arr
    is inputted, the default is zero
    '''
    if arr is None:
        arr = np.zeros(CALENDAR_SIZE)

    df = pd.DataFrame(arr, index = range(CALENDAR_SIZE[0]), columns=dates)
    return df

def traffic_data(dates, CALENDAR_SIZE):
    traffic_weights = []
    for date in dates:
        day = findDay(date)
        traffic_weights.append(traffic_boston[day])
    traffic = np.asarray(traffic_weights)
    traffic = np.transpose(traffic)
    mean_traffic = np.mean(traffic)
    std_traffic = np.std(traffic)
    normalized = (traffic - mean_traffic) / std_traffic
    return normalized

def events_data(CALENDAR_SIZE): 
    events = (np.random.normal(0, 1, CALENDAR_SIZE) > 1.5).astype(float)
    return events


def weighted_average(traffic, events, weights, CALENDAR_SIZE, hours = 20, mask = None):
    weighted_av = traffic #weights['traffic'] * traffic + weights['events'] * events
    weighted_av = weighted_av - np.min(weighted_av)
    if not mask is None:
        weighted_av[mask] = np.min(weighted_av) - 1
        
    weighted_av_thresh = weighted_av >= np.quantile(weighted_av, 1 - hours/(CALENDAR_SIZE[0] * CALENDAR_SIZE[1]))
    
    return weighted_av_thresh, weighted_av

def edge_link(wav_thresh): 
    wav_thresh[1: wav_thresh.shape[0] - 1] = wav_thresh[1: wav_thresh.shape[0]-1] | (wav_thresh[2:wav_thresh.shape[0]] & wav_thresh[:wav_thresh.shape[0] - 2])
    return wav_thresh
        

def simple_model(weights, mask, hours, CALENDAR_SIZE, dates, verbose = False): 
    np.random.seed(8)
    #get traffic data 
    traffic = traffic_data(dates, CALENDAR_SIZE)
    #return traffic #FOR DEBUGGING
    #get events data
    events = events_data(CALENDAR_SIZE)   
    #get weighted average traffic
    wav_t, wav = weighted_average(traffic, events, weights, CALENDAR_SIZE,hours = hours, mask=mask)
    #edge link    
    
    return edge_link(wav_t)