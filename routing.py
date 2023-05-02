from haversine import haversine,Unit
import pandas as pd
import numpy as np

df = pd.read_csv('result.csv').drop(columns=["Unnamed: 0"]).astype(int)
station = pd.read_csv('station_info.csv')
station.columns=["id","name","area","capacity","lat","lng"]



station = station[station['id'].isin(list(df['id']))]
station.reset_index(inplace=True,drop=True)

distance_matrix = np.zeros((51,51))

for row in station.index:
    for col in range(row,len(station)):
        if row == col:
            distance_matrix[row][col]=0
        else:
            coords_1 = (station.loc[row]["lat"],station.loc[row]["lng"])
            coords_2 = (station.loc[col]["lat"],station.loc[col]["lng"])
            distance_matrix[row][col] = haversine(coords_1,coords_2)*1000
            distance_matrix[col][row] = haversine(coords_1,coords_2)*1000


print(distance_matrix)