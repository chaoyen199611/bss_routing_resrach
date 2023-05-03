from haversine import haversine,Unit
import pandas as pd
import numpy as np
from mip import Model, xsum, minimize, BINARY,INTEGER

df = pd.read_csv('result.csv').drop(columns=["Unnamed: 0"]).astype(int)
station = pd.read_csv('station_info.csv')
station.columns=["id","name","area","capacity","lat","lng"]

print(df)

vehicle_num = 2
time_interval = 6
# initial inventory
q = np.zeros((vehicle_num))
# truck capacity
Qv = 20

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

y_plus = np.zeros((vehicle_num,time_interval,len(station)),dtype=int)
y_minus = np.zeros((vehicle_num,time_interval,len(station)),dtype=int)

#x = np.zeros((vehicle_num,time_interval,len(station),len(station)),dtype=bool)
#d+ and d- both set to 357 meters
d_minus = 357
d_plus = 357

model = Model()

x = [[[[model.add_var(var_type = BINARY) for j in range(len(station)+1)]for i in range(len(station))]for t in range(time_interval)]for v in range(vehicle_num)]
y_plus = [[[model.add_var(var_type = INTEGER) for i in range(len(station))]for t in range(time_interval)]for v in range(vehicle_num)]
y_minus = [[[model.add_var(var_type = INTEGER) for i in range(len(station))]for t in range(time_interval)]for v in range(vehicle_num)]

model.objective = minimize(xsum(distance_matrix[i][j]*x[v][t][i][j] for i in range(len(station)) for j in range(len(station)) for t in range(time_interval) for v in range(vehicle_num))
                        +xsum(d_minus*y_minus[v][t][i] for i in range(len(station)) for t in range(time_interval) for v in range(vehicle_num))
                        +xsum(d_plus*y_plus[v][t][i] for i in range(len(station)) for t in range(time_interval) for v in range(vehicle_num)))

#model constraint
print(df["start"][0])
for i in range(len(station)):
    model += (xsum(y_plus[v][t][i]-y_minus[v][t][i] for t in range(time_interval) for v in range(vehicle_num))+df["start"][i])>= df["smin"][i]
    model += (xsum(y_plus[v][t][i]-y_minus[v][t][i] for t in range(time_interval) for v in range(vehicle_num))+df["start"][i])<= df["smax"][i]
    model += xsum(x[v][t][i][j] for j in range(len(station)+1) for t in range(1,time_interval) for v in range(vehicle_num)) <= xsum(x[v][t-1][j][i] for j in range(len(station)) for t in range(1,time_interval) for v in range(vehicle_num))
    model += xsum(y_minus[v][t][i] for t in range(time_interval) for v in range(vehicle_num)) <= df["start"][i]
    model += xsum(y_plus[v][t][i] for t in range(time_interval) for v in range(vehicle_num)) <= df["capacity"][i] - df["start"][i]

    for t in range(time_interval):
        for v in range(vehicle_num):
            model += (xsum(x[v][t][i][j] for j in range(len(station)+1))*Qv) >= y_minus[v][t][i]
            model += (xsum(x[v][t][j][i] for j in range(len(station)))*Qv) >= y_plus[v][t][i]

for v in range(vehicle_num):
    for g in range(time_interval):
        model += (xsum(y_minus[v][t][i] - y_plus[v][t][i]for i in range(len(station)) for t in range(g)) + q[v])>=0
        model += (xsum(y_minus[v][t][i] - y_plus[v][t][i]for i in range(len(station)) for t in range(g)) + q[v])<=Qv


model += xsum(x[v][1][i][j] for v in range(vehicle_num) for i in range(len(station)) for j in range(len(station)+1)) <=1
model += xsum(x[v][t][i][i] for v in range(vehicle_num) for i in range(len(station)) for t in range(time_interval)) == 0

model.optimize()

print(x[0][0][1][0])
