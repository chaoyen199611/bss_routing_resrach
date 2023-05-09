import sys
from fetchdata import fetchdata

class Agent:
    def __init__(self,start_date,end_date,start_time,end_time,area):
        self.start_date = start_date
        self.end_date = end_date
        self.start_time = start_time
        self.end_time = end_time
        self.targetarea=area.split(",")
        self.status()

    def status(self):
        print("start_date:{}\nend_date:{}\ntime_interval:{}\ntarget_area:{}".format(self.start_date,\
                                                                    self.end_date,[self.start_time,self.end_time],self.targetarea))


if __name__=='__main__':
    agent = Agent(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5])
    fetchdata(agent.start_date,agent.end_date,agent.start_time,agent.end_time,agent.targetarea)
    