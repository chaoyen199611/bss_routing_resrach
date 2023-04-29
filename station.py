import pandas as pd


pickup_service_req = 0.85
return_service_req = 0.85


df = pd.read_csv('station_info.csv')
df.columns=["id","name","area","total","lat","lng"]

station_df = pd.read_csv('station_condition.csv')
station_df.columns=["id","recordid","bike","recordtime","free","active","last"]

trip = pd.read_csv('test.csv')
trip.columns=["start_time","end_time","startid","endid"]

station_df.drop(columns=["recordid"],inplace = True)


target_df = df.loc[((df['area'] == '新興區') | (df['area'] == '鹽埕區'))]

id_list = list(target_df["id"])
print(id_list)

# station_df = station_df.loc[station_df.id not in id_list]
station_df = station_df [station_df ['id'].isin(id_list)]
station_df.to_csv('station_info_test.csv')
trip = trip[trip["startid"].isin(id_list) | trip["endid"].isin(id_list)]

station_df.reset_index(inplace=True,drop=True)

result = pd.DataFrame()
result["capacity"]=[12]
station_df["day"] = pd.DatetimeIndex(station_df["recordtime"]).day
station_df["hour"] = pd.DatetimeIndex(station_df["recordtime"]).hour
station_df["minute"] = pd.DatetimeIndex(station_df["recordtime"]).minute
station_df["capacity"] = station_df["bike"]+station_df["free"]
station_df = station_df[(station_df["day"]==1) & (station_df["id"]==501201002)]
trip.reset_index(inplace=True,drop=True)
print(station_df)


trip_start = trip[trip["startid"]==501201002]
trip_end = trip[trip["endid"]==501201002]

trip_start.to_csv("trip_start_test.csv")
trip_end.to_csv("trip_end_test.csv")
station_df.to_csv('test_station.csv')

print(trip_start)