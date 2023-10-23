# %%

# Libraries
from TCtools import *
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats
from matplotlib import ticker
import json

plt.rcParams['font.sans-serif'] ='Arial'
# %% 整合处理

titles=['US Air Traffic TN','Hospital TN','PhD Exchange TN','Workplace TN',
'Primary School TN','Highschool TN','Village TN','SFHH TN']
titles2=['US Air Traffic TN','Hospital TN','PhD Exchange TN','Workplace TN',
'Primary School TN','High school TN','Village TN','SFHH TN']

TCi_dt_list=[]
ki_list=[]
dt_list_list=[]
title=[]
# titles=['Primary School TN']
for title_data in titles:
    with open(title_data+"_dt.json") as f_obj:
	    df2 = json.load(f_obj)  #读取文件
    TCi_dt_list.append(df2["TCi_dt"])
    ki_list.append(df2["ki"])  
    dt_list_list.append(df2['dt_list'])
    title.append(df2['title'])


# %%
size=24
fig=plt.figure(figsize=(24,12))
for i in range(8):
    ax=plt.subplot2grid((60,96),((i//4)*25+4,(i%4)*21),colspan=20,rowspan=20)
    a=list(np.vstack((np.array([ki_list[i]]),TCi_dt_list[i])).T)
    a.sort(key=lambda x:x[0],reverse=True)
    TCi_dt_sort=np.array(a)[:,1:]
    ax=sns.heatmap(data=TCi_dt_sort,cmap='jet',cbar=False,yticklabels=[],xticklabels=dt_list_list[i])
    ax.set_title(titles2[i][:-3],fontsize=size)
    x_list=[dt_list_list[i][0],dt_list_list[i][3],dt_list_list[i][6],dt_list_list[i][9]] 
    ax.set_xticks([0.5,3.5,6.5,9.5])
    ax.set_xticklabels(x_list,ha='center',fontsize=size,rotation=0)
        
    if (i==0) or (i==4):
        ax.set_ylabel('$k_{min}$                  $k_{max}$',fontsize=size)
    if i>=4:
        if (i==5) or (i==7):
            ax.set_xlabel("$\Delta t$",fontsize=size,labelpad=10)
        else:
            ax.set_xlabel("$\Delta t$",fontsize=size,labelpad=10)
position=fig.add_axes([0.81, 0.30,0.012 ,0.45])
cbar=fig.colorbar(ax.collections[0].colorbar,cax=position,shrink=0.8,cmap='jet')
cbar.ax.tick_params(labelsize=size)
cbar.set_ticks([0.00,0.25,0.50,0.75,1.00])
cbar.ax.set_title('$TC$',fontsize=size,x=2.7,y=1.08)
plt.savefig("TC-dt.pdf",dpi=50,format='pdf')

# %%
df2
# %%
