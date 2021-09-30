import pickle
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
from matplotlib.pyplot import figure
with open('stackplot.pkl', 'rb') as f:
    # (145, 125)
    q1=0
    q2=-1
    a,b,c = pickle.load(f)
    fig, ax = plt.subplots(2,3)
    # print(type(c[0]))
    ax[0,0].imshow(b[q1], cmap=cm.Greys_r)
    x = [62, 125]
    y = [73, 73]
    ax[0,1].plot(x, y, color="red", linewidth=1)
    ax[0,1].imshow(a[q1], cmap=cm.Greys_r)
    x = np.arange(62, 95)
    y = a[q1][73][62:95]
    # print(a[0].shape)
    ax[0,2].plot(x,y,color='blue')
    ax[0,2].plot(65, a[q1][73][65], marker='o', color="black")
    ax[0,2].plot(70, a[q1][73][70], marker='o', color="black")
    ax[0,2].plot(80, a[q1][73][80], marker='o', color="black")



    ax[1,0].imshow(b[q2], cmap=cm.Greys_r)
    x = [62, 125]
    y = [73, 73]
    ax[1,1].plot(x, y, color="red", linewidth=1)
    ax[1,1].imshow(a[q2], cmap=cm.Greys_r)
    x = np.arange(62, 95)
    y = a[q2][73][62:95]
    ax[1,2].plot(x,y,color='blue')
    ax[1,2].plot(65, a[q2][73][65], marker='o', color="black")
    ax[1,2].plot(70, a[q2][73][70], marker='o', color="black")
    ax[1,2].plot(80, a[q2][73][80], marker='o', color="black")
    fig.set_size_inches(18.5, 10.5)
    plt.savefig('plot.png',dpi=100)

