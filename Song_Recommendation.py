#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 21:15:54 2021

@author: rg
"""

import Camera
import pandas as pd
import random

def song_recommendations():
    emotion = Camera.snapshot()
    print(emotion)
    csv_name = "Song_Names/" + emotion + ".csv"
    df = pd.read_csv(csv_name)
    data = df.values.tolist()
    length = len(data)
    
    r = random.sample(range(0,length), 10)
    song_name = []
    songs = []
    for i in range(10):
        songs.append(str(data[r[i]]))
        song_name.append(songs[i].split('-')[0])
        song_name[i] = song_name[i].strip("['")
        
    return song_name
    
    
