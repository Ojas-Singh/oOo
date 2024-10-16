import pickle
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
from matplotlib.pyplot import figure
import math
import FKA
from matplotlib.cm import coolwarm,plasma


mpl.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 18
plt.rcParams['axes.linewidth'] = 2
plt.rcParams["figure.figsize"] = (10,6)
x1 = np.linspace(0,12,100) 
x2 = np.linspace(40,129,100)
c1='#DEE1DD'
plt.fill_between(x1, 0, 1300, facecolor=c1, alpha=0,label='1 pN', zorder=-20)
plt.fill_between(x2,0, 1300, facecolor=c1, alpha=0,label='9 pN', zorder=-20)
plt.xlim([8, 129])
plt.ylim([320, 1120])
#plt.yaxis.set_major_locator(mpl.ticker.MultipleLocator(200))
with open('stack.pkl', 'rb') as f:
    a = pickle.load(f)
    l= []
    for i in a[0]:
        l.append(i[0])
    cmap = mpl.cm.plasma
    norm = mpl.colors.Normalize(vmin=0, vmax=2000)
    for k in range(0,len(l),1):
        xp = [i for i in range(10,len(l[k])+10)]
        # plt.plot(xp,l[k],linewidth=0.5,c=color_dict1.get('hex')[k],alpha=0.7)
        plt.plot(xp,l[k],linewidth=0.75,c=plasma(k/len(l)),alpha=0.8)
    plt.plot()
    plt.xlabel('Radial Distance (px)', labelpad=10)
    plt.ylabel('Intensity', labelpad=10)
    plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
             orientation='vertical', label='Stack Position (nm)')
    plt.savefig('output/stackplot.png',dpi=1200)
    plt.savefig("output/stackplot.svg", format="svg")
    plt.show()
    # plt.savefig('output/stackplot.png',dpi=450)