import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def mip_routing_result():
    station_num=57
    df = pd.read_csv("data/service_level_result.csv")
    df.drop(df[(df["smin"]==0)&(df["smax"]==0)].index,inplace=True)
    df.reset_index(inplace=True,drop=True)
    
    # unservice = df.loc[(df["smin"]>df["start"])|(df["smax"]<df["start"])]
    station_condition = pd.read_csv("data/station_condition.csv").tail(station_num)
    station_condition.reset_index(inplace=True,drop=True)
    df["after3"] = station_condition["bike"]
    
    print(df)
    


if __name__ =='__main__':
    mip_routing_result()