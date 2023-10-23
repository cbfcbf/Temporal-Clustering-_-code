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
import random
# %%
size=30
dt=10
w_min=1
k_min=1
fig=plt.figure(figsize=(18, 9))

df=pd.read_csv('USAL_TN_monthly_2012_2020_shuffled5.txt', header=None, sep="\s+")
title='US Air Traffic TN_shuffled $P(L)$'
# ordered_IDs = pd.read_csv('../US Air Traffic TN/ordered_airport_ID.txt', header=None, sep="\s+")
df.columns=(['t','i','j'])
Gt,G_static,G_w,al_agg,nodes,_=init_net(USAL_TN_df=df)
title2="Uniform time permutation"


G_dt,_=G_dt_generator2(dt=dt,Gt=Gt)    
C_list,TCi,ki,kwi,C_list_total,TCi_total,kwi0,ki0=find_index2(G_static=G_static,G_t=G_dt,G_w=G_w,k_min=k_min,w_min=w_min)
edge_num=[]
for g in G_dt:
    edge_num.append(len(g.edges()))
edge_num=np.array(edge_num)
p=round(edge_num.mean()/len(G_static.edges()),2)

ax=plt.subplot2grid((25,50),(0,0),colspan=20,rowspan=20)
ax.scatter(C_list,TCi,c=kwi,s=np.array(ki)/max(ki)*600,cmap="jet",alpha=0.5)
ax.set_xlim((-0.05, 1.05))
ax.set_ylim((-0.05, 1.05))
ax.set_title(title2,fontsize=size)
#plt.title("Time permution keeping the distribution of $w$",fontsize=24)
# plt.title("Time permution keeping the strcture of static network",fontsize=24)

ax.plot([0,1],[0,1], color='black',linestyle=':',linewidth=5)
ax.plot([0,1],[0,p], color='r',linestyle='dashed',linewidth=5) 

ax.set_xlabel("$C$",fontsize=size)
ax.set_xticks([0.0,0.5,1.0],fontsize=size)
ax.set_ylabel("$TC$",fontsize=size)
ax.set_yticks([0.0,0.5,1.0],fontsize=size)
ax.tick_params(labelsize=size)

df=pd.read_csv('USAL_TN_monthly_2012_2020_shuffled6.txt', header=None, sep="\s+")
title='US Air Traffic TN_shuffled $P(p(w))$'
# ordered_IDs = pd.read_csv('../US Air Traffic TN/ordered_airport_ID.txt', header=None, sep="\s+")
df.columns=(['t','i','j'])
Gt,G_static,G_w,al_agg,nodes,_=init_net(USAL_TN_df=df)
title1="Weighted time permutation"

G_dt,_=G_dt_generator2(dt=dt,Gt=Gt)    
C_list,TCi,ki,kwi,C_list_total,TCi_total,kwi0,ki0=find_index2(G_static=G_static,G_t=G_dt,G_w=G_w,k_min=k_min,w_min=w_min)
edge_num=[]
for g in G_dt:
    edge_num.append(len(g.edges()))
edge_num=np.array(edge_num)
p=round(edge_num.mean()/len(G_static.edges()),2)
ax=plt.subplot2grid((25,50),(0,23),colspan=25,rowspan=20)
im=ax.scatter(C_list,TCi,c=kwi,s=np.array(ki)/max(ki)*600,cmap="jet",alpha=0.5)
ax.set_xlim((-0.05, 1.05))
ax.set_ylim((-0.05, 1.05))
ax.set_title(title1,fontsize=size)
# plt.title("Time permution keeping the distribution of $w$",fontsize=24)
# plt.title("Time permution keeping the strcture of static network",fontsize=24)

ax.plot([0,1],[0,1], color='black',linestyle=':',linewidth=5)
ax.plot([0,1],[0,p], color='r',linestyle='dashed',linewidth=5) 
ax.set_xlabel("$C$",fontsize=size)
ax.set_xticks([0.0,0.5,1.0],fontsize=size)
ax.set_yticks([])

ax.tick_params(labelsize=size)

cbar = plt.colorbar(im)
cbar.set_ticks([])
cbar.set_label("$k^w_{\min}$                             $k^w_{\max}$",fontsize=size)
plt.savefig("randomized example.pdf",format='pdf')

# %%
