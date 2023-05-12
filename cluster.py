import pandas as pd
import numpy as np


df = pd.read_csv('data/service_level_result.csv')
# begin = pd.read_csv("result.csv")

print(df)

df.columns=["id","rid","bike","recordtime","freespace","active","idd","area"]


new_row = pd.DataFrame({'id':501201030, 'rid':111, 'bike':'3', 'recordtime':1800,'freespace':12,'active':1,"idd":11,"area":11}, index=[0])
df = pd.concat([new_row,df.loc[:]]).reset_index(drop=True)

df.drop(columns=["rid","recordtime","idd","area"],inplace=True)
begin["after3"] = df["bike"]
begin["active"] = df["active"]
begin.drop(columns=["Unnamed: 0"],inplace=True)
begin.drop(begin[begin["active"]==0].index,inplace=True)
begin.drop(columns=["active"],inplace=True)
begin["diff"] = begin["after3"].astype(int) - begin["start"]
pick_drop = [4,-2,0,0,0,0,0,0,-10,0,0,0,0,0,0,0,0,-1,0,-8,0,0,4,0,0,7,0,1,2,0,2,0,-6,0,0,0,0,0,-6,0,0,0,5,0,1,2,0,0,0,0,0]

vehicle1 = [17,27,19,0,1,28,22,8,25]
vehicle2 = [38,47,30,32,45,42,44]


pickdrop_df=pd.DataFrame(pick_drop)
begin.reset_index(inplace=True,drop=True)
begin["pick & drop"] = pickdrop_df[0]


qdf = begin[(begin["after3"].astype(int)+begin["pick & drop"]>begin["smax"]) | (begin["after3"].astype(int)+begin["pick & drop"]<begin["smin"])]

print("="*70)
print('{:^68s}'.format("stations need to be rebalance"))
print("="*70)
print(begin[(begin["start"]<begin["smin"]) | (begin["start"]>begin["smax"])])

print("="*70)
print('{:^68s}'.format("can't reach service requirements after rebalance"))
print("="*70)

print(qdf)
print("="*70)
print('{:^68s}'.format("unbalance at the beginning"))
print("="*70)
print(begin[(begin["start"]==begin["capacity"]) | (begin["start"]==0)])
print("="*70)
print('{:^68s}'.format("unbalance after 3 hours"))
print("="*70)
print(begin[(begin["after3"].astype(int)==begin["capacity"]) | (begin["after3"].astype(int)==0)])


print("調度站點：{}".format([begin.loc[8]["id"].astype(int),begin.loc[38]["id"].astype(int),begin.loc[44]["id"].astype(int)]))
print("維修站點：{}".format([begin.loc[44]["id"].astype(int)]))
print("因開始狀態已達標準為調度，三小時後未達標準：{}".format([begin.loc[6]["id"].astype(int),begin.loc[7]["id"].astype(int),begin.loc[35]["id"].astype(int),begin.loc[47]["id"].astype(int)]))
# print(begin)