from haversine import haversine,Unit
import pandas as pd
import numpy as np
from mip import Model, xsum, minimize, BINARY,INTEGER,ConstrList
import matplotlib.pyplot as plt

df = pd.read_csv('result.csv').drop(columns=["Unnamed: 0"]).astype(int)

station = pd.read_csv('station_info.csv')
station.columns=["id","name","area","capacity","lat","lng"]

df.drop(df[(df["smin"]==0)&(df["smax"]==0)].index,inplace=True)
df.reset_index(inplace=True,drop=True)

original = df.copy();

vehicle_num = 2
# time_interval is based on how many insufficient stations
time_interval = 20
# initial inventory
q = np.zeros((vehicle_num))
# truck capacity
Qv = 14

station = station[station['id'].isin(list(df['id']))]

station.reset_index(inplace=True,drop=True)


S = set(df.index)
N = S.copy()
N.add(-1)


distance_matrix = np.zeros((len(S),len(N)))

for row in range(len(S)):
    for col in range(row,len(N)):
        if row == col:
            distance_matrix[row][col]=0
        else:
            if col == len(N)-1:
                distance_matrix[row][col] = 0
            else:
                coords_1 = (station.loc[row]["lat"],station.loc[row]["lng"])
                coords_2 = (station.loc[col]["lat"],station.loc[col]["lng"])
                distance_matrix[row][col] = haversine(coords_1,coords_2)*1000
                distance_matrix[col][row] = haversine(coords_1,coords_2)*1000

d_minus = 357
d_plus = 357

model = Model()
h =0
x = [[[[model.add_var(var_type = BINARY) for i in range(len(S))]for j in range(len(N))]for t in range(time_interval)]for v in range(vehicle_num)]
y_plus = [[[model.add_var(var_type = INTEGER) for i in range(len(S))]for t in range(time_interval)]for v in range(vehicle_num)]
y_minus = [[[model.add_var(var_type = INTEGER) for i in range(len(S))]for t in range(time_interval)]for v in range(vehicle_num)]

# for v in range(vehicle_num):
#     model.objective = minimize(xsum(distance_matrix[i][j]*x[v][t][i][j] for i in range(len(S)) for j in range(len(S)) for t in range(time_interval))
#                         +xsum(d_minus*y_minus[v][t][i] for i in range(len(S)) for t in range(time_interval))
#                         +xsum(d_plus*y_plus[v][t][i] for i in range(len(S)) for t in range(time_interval)))

for v in range(vehicle_num):
    h += (xsum(distance_matrix[i][j]*x[v][t][i][j] for i in range(len(S)) for j in range(len(S)) for t in range(time_interval))
        +xsum(d_minus*y_minus[v][t][i] for i in range(len(S)) for t in range(time_interval))
        +xsum(d_plus*y_plus[v][t][i] for i in range(len(S)) for t in range(time_interval)))

model.objective = minimize(h)
#model constraint

for i in range(len(S)):
    model += (xsum(y_plus[v][t][i]-y_minus[v][t][i] for t in range(time_interval) for v in range(vehicle_num))+df.loc[i]["start"])>= df.loc[i]["smin"]
    model += (xsum(y_plus[v][t][i]-y_minus[v][t][i] for t in range(time_interval) for v in range(vehicle_num))+df.loc[i]["start"])<= df.loc[i]["smax"]

for v in range(vehicle_num):
    model += xsum(x[v][0][j][i] for i in range(len(S)) for j in range(len(N))) <=1

for i in range(len(S)):
    for t in range(1,time_interval):
        for v in range(vehicle_num):
            model += xsum(x[v][t][j][i] for j in range(len(N))) <= xsum(x[v][t-1][i][j] for j in range(len(S)))

model += xsum(x[v][t][i][i] for i in range(len(S)) for t in range(time_interval) for v in range(vehicle_num)) == 0  

for v in range(vehicle_num):
    for t in range(time_interval):
        for i in range(len(S)):        
            model += (xsum(x[v][t][j][i] for j in range(len(N)))*Qv) >= y_minus[v][t][i]
            model += (xsum(x[v][t][i][j] for j in range(len(S)))*Qv) >= y_plus[v][t][i]

for i in range(len(S)):
    model += xsum(y_minus[v][t][i] for t in range(time_interval) for v in range(vehicle_num)) <= df.loc[i]["start"]
    model += xsum(y_plus[v][t][i] for t in range(time_interval) for v in range(vehicle_num)) <= df.loc[i]["capacity"] - df.loc[i]["start"]

for v in range(vehicle_num):
    for g in range(time_interval):
        model += (xsum(y_minus[v][t][i] - y_plus[v][t][i]for i in range(len(S)) for t in range(g+1)) + q[v])>=0
        model += (xsum(y_minus[v][t][i] - y_plus[v][t][i]for i in range(len(S)) for t in range(g+1)) + q[v])<=Qv


model.optimize()
#


# for j in range(len(station)):
#     for i in range(len(station)):
#         print(x[0][0][j][i].x,end=' ')
    
#     print('\n')

# print(x[0][0][27][17].x)
# print(y_plus[0][1][27].x)
for t in range(time_interval):
    for i in range(len(station)):
        for v in range(vehicle_num):
            if(y_plus[v][t][i].x!=0):
                print("vehicle_num : {} time_interval : {} station : {} drop_off : {}".format(v,t,i,y_plus[v][t][i].x))
                df.loc[i]["start"]=df.loc[i]["start"]+y_plus[v][t][i].x
            if(y_minus[v][t][i].x!=0):
                print("vehicle_num : {} time_interval : {} station : {} pick_up : {}".format(v,t,i,y_minus[v][t][i].x))
                df.loc[i]["start"]=df.loc[i]["start"]-y_minus[v][t][i].x



for t in range(time_interval):
    for v in range(vehicle_num):
        for i in range(len(station)):
            for j in range(len(station)):
                if x[0][t][j][i].x!= False:
                    print("vehicle_num : {} time_interval : {} arc i{} to j{}".format(v,t,i,j))


x_cord = list(original.index)
y_cord = list(original["start"])
y_cord2 = list(df["start"])

y_smin = list(df["smin"])
y_smax = list(df["smax"])





# plt.plot(x_cord,y_smin,'b')
# plt.plot(x_cord,y_smax,'b')



plt.fill_between(x_cord, y_smin, y_smax)
plt.plot(x_cord,y_cord,'black')
plt.plot(x_cord,y_cord2,'r')

plt.show()


            

    
