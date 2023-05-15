import pandas as pd
import numpy as np
from haversine import haversine,Unit
from mip import Model, xsum, minimize, BINARY,INTEGER,ConstrList,maximize
import matplotlib.pyplot as plt


vehicle_num = 2
truck_capacity = 14

df = pd.read_csv('data/service_level_result.csv').drop(columns=["Unnamed: 0"]).astype(int)
station = pd.read_csv('data/station_info.csv')


df.drop(df[(df["smin"]==0)&(df["smax"]==0)].index,inplace=True)
df.reset_index(inplace=True,drop=True)

s_plus = np.zeros(len(df))
s_minus = np.zeros(len(df))
spluslist = list(df["smin"]-df["start"])
sminuslist = list(df["start"]-df["smax"])

for i in range(len(df)):
    s_plus[i] = max(spluslist[i],0)
    s_minus[i] = max(sminuslist[i],0)

insufficent = (df["start"]<df["smin"]) | (df["start"]>df["smax"])

start = list(df["start"])
smin = list(df["smin"])
smax = list(df["smax"])


# initial inventory
q = np.zeros((vehicle_num))
# truck capacity
Qv = truck_capacity

station = station[station['id'].isin(list(df['id']))]

station.reset_index(inplace=True,drop=True)


S = set(df.index)


distance_matrix = np.zeros((len(S),len(S)))

for row in range(len(S)):
    for col in range(row,len(S)):
        if row == col:
            distance_matrix[row][col]=0
        else:
            coords_1 = (station.loc[row]["lat"],station.loc[row]["lng"])
            coords_2 = (station.loc[col]["lat"],station.loc[col]["lng"])
            distance_matrix[row][col] = haversine(coords_1,coords_2)*1000
            distance_matrix[col][row] = haversine(coords_1,coords_2)*1000

d_minus = 357
d_plus = 357



model = Model()

h = [model.add_var(var_type= INTEGER) for v in range(vehicle_num)]
z = [[model.add_var(var_type= BINARY) for i in range(len(S))] for v in range(vehicle_num)] 

total=0

for v in range(vehicle_num):
    total += h[v]
    
for i in df[insufficent].index:
    model += xsum(z[v][i] for v in range(vehicle_num)) == 1

for i in df[~insufficent].index:
    model += xsum(z[v][i] for v in range(vehicle_num)) <= 1

for v in range(vehicle_num):
    model += (q[v]+xsum(start[i]*z[v][i] for i in range(len(S))) >= xsum(smin[i]*z[v][i] for i in range(len(S))))
    model += (-(Qv-q[v])+xsum(start[i]*z[v][i] for i in range(len(S))) <= xsum(smax[i]*z[v][i] for i in range(len(S))))
    


for v in range(vehicle_num):
    for i in range(len(S)):
        model += (h[v] >= xsum(distance_matrix[i][j] * (z[v][i]+z[v][j]-1) for j in range(len(S)))+xsum((d_plus*s_plus[j]+d_minus*s_minus[j])*z[v][j] for j in range(len(S))))
        model += (h[v] >= xsum(distance_matrix[i][j] * (z[v][i]+z[v][j]-1) for j in range(len(S)))+xsum((d_plus*s_plus[j]+d_minus*s_plus[j])*z[v][j] for j in range(len(S)))-d_minus*q[v])
        model += (h[v] >= xsum(distance_matrix[i][j] * (z[v][i]+z[v][j]-1) for j in range(len(S)))+xsum((d_plus*s_minus[j]+d_minus*s_minus[j])*z[v][j] for j in range(len(S)))-d_plus*(Qv-q[v]))


model.objective = minimize(total)

model.optimize()

vehicle1_cluster=[]
vehicle2_cluster=[]

for i in range(len(S)):
    if z[0][i].x==1:
        vehicle1_cluster.append(i)
    if z[1][i].x==1:
        vehicle2_cluster.append(i)
        
vehicle1_cluster_df = station.loc[vehicle1_cluster]

vehicle1_cluster_df.drop_duplicates(inplace = True)

vehicle2_cluster_df = station.loc[vehicle2_cluster]
vehicle2_cluster_df.drop_duplicates(inplace = True)

fig, (ax1, ax2) = plt.subplots(1, 2)

ax1.scatter(x=vehicle1_cluster_df['lng'], y=vehicle1_cluster_df['lat'])
ax1.scatter(x=vehicle2_cluster_df['lng'], y=vehicle2_cluster_df['lat'])


result = np.load("data/np_save.npy")   

check = result[0][0][0][0]
check2 = result[1][0][0][0]
#np shape(vehicle num, time interval, drop & pickup, arc)
for i in range(len(result[0])):
    if result[0][i][0][0]!=0 and result[0][i][0][1]!=0:
        check = np.concatenate((check,result[0][i][0][1]),axis=None)
    if result[1][i][0][0]!=0 and result[1][i][0][1]!=0:
        check2 = np.concatenate((check2,result[1][i][0][1]),axis=None)
print(check)
print(check2)
vehicle1_noncluster_df = station.loc[check]
vehicle1_noncluster_df.drop_duplicates(inplace = True)

vehicle2_noncluster_df = station.loc[check2]
vehicle2_noncluster_df.drop_duplicates(inplace = True)

ax2.scatter(x=vehicle1_noncluster_df['lng'], y=vehicle1_noncluster_df['lat'])
ax2.scatter(x=vehicle2_noncluster_df['lng'], y=vehicle2_noncluster_df['lat'])

ax2.set_xticks([])
ax2.set_yticks([])
ax1.set_xticks([])
ax1.set_yticks([])

ax1.set_title('cluster')
ax1.set_xlabel("longitude")
ax1.set_ylabel("latitude")

ax2.set_title('non cluster')
ax2.set_xlabel("longitude")
ax2.set_ylabel("latitude")

plt.tight_layout()
plt.show()
