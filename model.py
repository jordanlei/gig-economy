import numpy as np
import pandas as pd
from datetime import date, timedelta, datetime


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

def traffic_data(CALENDAR_SIZE): 
    noise = np.random.normal(0, 1, CALENDAR_SIZE)
    x = np.arange(CALENDAR_SIZE[1])
    y = np.arange(CALENDAR_SIZE[0])
    _, yv = np.meshgrid(x, y)
    traffic = noise*0.2 + (-1* np.cos(yv/1.8))
    mean_traffic = np.mean(traffic)
    std_traffic = np.std(traffic)
    normalized = (traffic - mean_traffic) / std_traffic
    return normalized

def events_data(CALENDAR_SIZE): 
    events = (np.random.normal(0, 1, CALENDAR_SIZE) > 1.5).astype(float)
    return events


def weighted_average(traffic, events, weights, CALENDAR_SIZE, hours = 20, mask = None):
    weighted_av = weights['traffic'] * traffic + weights['events'] * events
    weighted_av = weighted_av - np.min(weighted_av)
    if not mask is None:
        weighted_av[mask] = np.min(weighted_av) - 1
        print("Here")
        
    weighted_av_thresh = weighted_av >= np.quantile(weighted_av, 1 - hours/(CALENDAR_SIZE[0] * CALENDAR_SIZE[1]))
    
    return weighted_av_thresh, weighted_av

def edge_link(wav_thresh): 
    wav_thresh[1: wav_thresh.shape[0] - 1] = wav_thresh[1: wav_thresh.shape[0]-1] | (wav_thresh[2:wav_thresh.shape[0]] & wav_thresh[:wav_thresh.shape[0] - 2])
    return wav_thresh
        

def simple_model(weights, mask, hours, CALENDAR_SIZE, verbose = False): 
    np.random.seed(8)
    #get traffic data
    traffic = traffic_data(CALENDAR_SIZE)
    #get events data
    events = events_data(CALENDAR_SIZE)   
    #get weighted average traffic
    wav_t, wav = weighted_average(traffic, events, weights, CALENDAR_SIZE,hours = hours, mask=mask)
    #edge link    
    return edge_link(wav_t)