import pickle
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
from matplotlib.pyplot import figure
import math
import FKA
from matplotlib.cm import coolwarm,plasma


mpl.rcParams['font.family'] = 'Cambria'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1
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
        plt.plot(xp,l[k],linewidth=0.5,c=plasma(k/len(l)),alpha=0.8)
    plt.plot()
    plt.xlabel('Radial Distance (px)', labelpad=10)
    plt.ylabel('Intensity', labelpad=10)
    plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
             orientation='vertical', label='Z (nm)')
    plt.savefig('output/stackplot.png',dpi=500)
    plt.show()
    # plt.savefig('output/stackplot.png',dpi=450)