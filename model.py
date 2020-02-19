import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import sklearn

CALENDAR_SIZE = (24, 7) #set calendar size to 24 by 7 

def generate_calendar_matrix(arr = None): 
    '''
    Generates a calendar matrix based off
    an optional numpy input array. If no arr
    is inputted, the default is zero
    '''
    if arr is None:
        arr = np.zeros(CALENDAR_SIZE)
    df = pd.DataFrame(arr, index = range(CALENDAR_SIZE[0]), columns= ["mon", "tues", "wed", "thur", "fri", "sat", "sun"])
    return df

def traffic_data(): 
    noise = np.random.normal(0, 1, CALENDAR_SIZE)
    x = np.arange(CALENDAR_SIZE[1])
    y = np.arange(CALENDAR_SIZE[0])
    _, yv = np.meshgrid(x, y)
    traffic = noise*0.2 + (-1* np.cos(yv/1.8))
    mean_traffic = np.mean(traffic)
    std_traffic = np.std(traffic)
    normalized = (traffic - mean_traffic) / std_traffic
    return normalized

def events_data(): 
    events = (np.random.normal(0, 1, CALENDAR_SIZE) > 1.5).astype(float)
    return events


def weighted_average(traffic, events, weights, hours = 20, mask = None):
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


        

def simple_model(weights, mask, hours, verbose = False): 
    np.random.seed(8)
    #get traffic data
    traffic = traffic_data()
    #get events data
    events = events_data()   
    #get weighted average traffic
    wav_t, wav = weighted_average(traffic, events, weights, hours = hours, mask=mask)
    #edge link    
    return edge_link(wav_t)