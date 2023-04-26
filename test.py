import mysql.connector
import pandas as pd
import csv
import datetime


connection = mysql.connector.connect(host='localhost',
                                    database='youbike',
                                    user='joeyhuang',
                                    password='open0813')

cursor = connection.cursor()

sql_select_query = "select * from triprecord where day(start_time) between 1 and 5 and time(start_time) between '06:00:00' and '08:59:59'"
station_condition = "select * from test where day(recordTime) between 1 and 5 and time(recordTime) between '06:00:00' and '08:59:59'"
station_info = "select * from stationinfo";
cursor.execute(sql_select_query)
records = cursor.fetchall()

fp = open('test.csv','w')
myfile = csv.writer(fp)
myfile.writerows(records)

cursor.execute(station_condition)
station_data = cursor.fetchall()
print(station_data)

fp = open('station_condition.csv','w')
myfile = csv.writer(fp)
myfile.writerows(station_data)

cursor.execute(station_info)
stationinfo = cursor.fetchall()

fp = open('station_info.csv','w')
myfile = csv.writer(fp)
myfile.writerows(stationinfo)


fp.close()
cursor.close()
connection.close()