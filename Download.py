#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 21:53:50 2021

@author: rg
"""

import Song_Recommendation
import urllib.request
import re
import youtube_dl

def download_song():
    song_list = Song_Recommendation.song_recommendations()
    for i in range(10):
        url = song_list[i].replace(" ", "+")
        tmp = "https://www.youtube.com/results?search_query=" + url        
        html = urllib.request.urlopen(tmp)
        video_ids = re.findall(r"watch\?v=(\S{11})", html.read().decode())
        # print(video_ids)
        
        down = "https://www.youtube.com/watch?v=" + video_ids[0]
        down_name = 'Songs/' + song_list[i].replace(" ", "_") + '.mp3'
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': down_name,
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
                }],
            }
        
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            ydl.download([down])    
    
    return song_list
