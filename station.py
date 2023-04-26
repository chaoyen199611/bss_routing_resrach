import pandas as pd


pickup_service_req = 0.85
return_service_req = 0.85


df = pd.read_csv('station_info.csv')
df.columns=["id","name","area","total","lat","lng"]

station_df = pd.read_csv('station_condition.csv')
station_df.columns=["id","recordid","bike","recordtime","free","active","last"]

trip = pd.read_csv('test.csv')
trip.columns=["start_time","end_time","startid","endid"]

station_df.drop(columns=["recordid","last"],inplace = True)


target_df = df.loc[((df['area'] == '新興區') | (df['area'] == '鹽埕區'))]

id_list = list(target_df["id"])

# station_df = station_df.loc[station_df.id not in id_list]
station_df = station_df [station_df ['id'].isin(id_list)]
trip = trip[trip["startid"].isin(id_list) | trip["endid"].isin(id_list)]

result = pd.DataFrame()
result["capacity"]=[12]

print(result)