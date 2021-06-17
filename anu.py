import time
import psutil
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
fig.show()
i = 0
x, y = [], []

while True:
    x.append(i)
    y.append(psutil.cpu_percent())
    
    ax.plot(x, y, color='b')
    plt.pause(0.05)
    fig.canvas.draw()
    plt.pause(0.05)
    ax.set_xlim(left=max(0, i-50), right=i+50)
    print(i)
    plt.pause(0.05)
    plt.show(block=False)
    time.sleep(0.1)
    i += 1
# from numpy import *
# from pylab import *
# x = linspace(-3, 3, 30)
# y = x**2
# plot(x, y)
# show()



# import random
# from itertools import count
# # import pandas as pd
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
# # multiprocessing .set_start_method('forkserver')
# import multiprocessing
# from multiprocessing import Pool, Queue
# plt.style.use('fivethirtyeight')

# x_vals = []
# y_vals = []

# index = count()
# data = [[],[]]

# def d(q):
#     def animate(i):
#         # data = pd.read_csv('data.csv')
#         # dat = data.clone()
#         dummylist=[]
#         for i in range(q.qsize()):
#                 dummylist.append(q.get())
#         x,y = dummylist

#         plt.cla()

#         plt.plot(x, y, label='Channel 1')

#         plt.legend(loc='upper left')
#         plt.tight_layout()
        


#     ani = FuncAnimation(plt.gcf(), animate, interval=1000)
#     plt.tight_layout()
#     plt.show(block=False)
# q= Queue()
# D = multiprocessing.Process(target=d, args=[q],daemon = True)
# D.start()
# i=0

# while True :

#     # data[0].append(i)
#     # data[1].append(random.random())
#     q.put((i,random.random()))
#     i+=1
# D.terminate()
