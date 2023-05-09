import mysql.connector
import pandas as pd
import csv
import datetime

def fetchdata(start_date,end_date,start_time,end_time,area):
    
    #raw triprecord from 4/1 to 4/10
    df = pd.read_csv("data/triprecord.csv")
    df.drop(columns=["Unnamed: 0"],inplace=True)
    station_info = pd.read_csv("data/station_info.csv")
    print(station_info)
    #station_df = station_df [station_info['id'].isin(id_list)]

    df["rent_time"]=pd.to_datetime(df["rent_time"])
    df["time"]=pd.to_datetime(df["time"])
    print(area)

    print(df["time"].dt.day)