import pandas as pd
import numpy as np
from kol import kolmogorov_forward_equation
import matplotlib.pyplot as plt

def service_level(station_num):
    station_info = pd.read_csv('data/station_condition.csv')
    station_info['capacity'] = station_info['bike']+station_info['free']
    cap=list(station_info['capacity'][:station_num ])
    start = list(station_info['bike'][:station_num ])
    station_list = list(station_info['id'].unique())
    print(len(station_list),station_num)
    result = np.zeros((station_num,5))
    trip_start = pd.read_csv("data/start_triprecord.csv")
    trip_end = pd.read_csv("data/end_triprecord.csv")
    print(station_num)
    for i in range(station_num):
        station = station_list[i]
        current_trip_start = trip_start[trip_start["rent_s_no"]==station]

        current_trip_end = trip_end[trip_end["s_no"]==station]
        mu = len(current_trip_start)/180
        lam = len(current_trip_end)/180


        result[i][0],result[i][1] = kolmogorov_forward_equation(mu,lam,cap[i])
        result[i][2] = station
        result[i][3] = cap[i]
        result[i][4] = start[i]


    df = pd.DataFrame(result, columns = ['smin','smax','id','capacity','start'])
    df.to_csv('data/service_level_result.csv')

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
    plt.savefig('plot/service_level_result.png')
    plt.close()


    
    


