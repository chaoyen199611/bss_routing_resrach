import pandas as pd
import numpy as np
from kol import kolmogorov_forward_equation
import matplotlib.pyplot as plt

# df = pd.read_csv("test_station.csv")
# trip_start= pd.read_csv("trip_start_test.csv")
# trip_end = pd.read_csv("trip_end_test.csv")
# trip_start.drop(columns=["Unnamed: 0"],inplace=True)
# trip_end.drop(columns=["Unnamed: 0"],inplace=True)

# df.drop(columns=["Unnamed: 0"],inplace=True)

# trip_start["hour"] = pd.DatetimeIndex(trip_start["start_time"]).hour
# trip_start["minute"] = pd.DatetimeIndex(trip_start["start_time"]).minute

# trip_end["hour"] = pd.DatetimeIndex(trip_end["end_time"]).hour
# trip_end["minute"] = pd.DatetimeIndex(trip_end["end_time"]).minute

# mu = np.zeros((3,4))
# lam = np.zeros((3,4))
# mu = len(trip_start)/12
# lam = len(trip_end)/12

# for i in range(3):
#     for j in range(1,4):
#         minute = j*15
#         mu[i][j] = len(trip_start[(trip_start["hour"]==(i+6)) & (trip_start["minute"]<=minute)])
#         lam[i][j] = len(trip_end[(trip_end["hour"]==(i+6)) & (trip_end["minute"]<=minute)])

station_info = pd.read_csv('station_info_test.csv')
station_info['capacity'] = station_info['bike']+station_info['free']
trip_record = pd.read_csv('test.csv')
trip_record.columns=["start_time","end_time","startid","endid"]
station_num = station_info['id'].nunique()
cap=list(station_info['capacity'][:51])
station_list = list(station_info['id'].unique())
result = np.zeros((station_num,2))
print(result[:,0])
print(station_num)
for i in range(station_num):
    station = station_list[i]
    trip_start = trip_record[trip_record["startid"]==station]
    trip_end = trip_record[trip_record["endid"]==station]
    mu = len(trip_start)/180
    lam = len(trip_end)/180
    print(station)

    result[i][0],result[i][1] = kolmogorov_forward_equation(mu,lam,cap[i])

df = pd.DataFrame(result, columns = ['smin','smax'])
df.to_csv('result.csv')

def hat_graph(ax, xlabels, values, group_labels):

    def label_bars(heights, rects):
        """Attach a text label on top of each bar."""
        for height, rect in zip(heights, rects):
            ax.annotate(f'{height}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 4),  # 4 points vertical offset.
                        textcoords='offset points',
                        ha='center', va='bottom')

    values = np.asarray(values)
    x = np.arange(values.shape[1])
    ax.set_xticks(x, labels=xlabels)
    spacing = 0.3  # spacing between hat groups
    width = (1 - spacing) / values.shape[0]
    heights0 = values[0]
    for i, (heights, group_label) in enumerate(zip(values, group_labels)):
        style = {'fill': False} if i == 0 else {'edgecolor': 'black'}
        rects = ax.bar(x - spacing/2 + i * width, heights - heights0,
                       width, bottom=heights0, label=group_label, **style)
        label_bars(heights, rects)


# initialise labels and a numpy array make sure you have
# N labels of N number of values in the array

smin_bound = result[:,0]
smax_bound = result[:,1]

fig, ax = plt.subplots()
hat_graph(ax, station_list, [smin_bound, smax_bound], ['smin_bound', 'smax_bound'])

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('statoin')
ax.set_ylabel('bikes')
ax.set_ylim(0, 50)
ax.set_title('stations service_level at 2023 April first to fifth, 6 a.m. - 9 a.m.')
ax.legend()

fig.tight_layout()
plt.show()
    
    


