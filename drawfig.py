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
import json
plt.rcParams['font.sans-serif'] ='Arial'
# %% 整合处理
titles=['US Air Traffic TN','Hospital TN','PhD Exchange TN','Workplace TN',
'Primary School TN','High School TN','Village TN','SFHH TN']
C_list,TCi,ki,kwi,Ci_kmin,ki0,TCi_kmin,Ci_wmin,TCi_wmin,title,k_min,w_min,dt,maxw,maxk,p_list=read_draw_data(titles)
# %% TC-C图
size=24
fig=plt.figure(figsize=(24, 12))
for i in range(8):
    if i%4==3:
        ax1=plt.subplot2grid((60,96),((i//4)*21+4,(i%4)*21),colspan=24,rowspan=20)
        im=ax1.scatter(C_list[i],TCi[i],c=kwi[i],s=np.array(ki[i])/max(ki[i])*300,cmap="jet",alpha=0.5)
        ax1.set_xlim((-0.05, 1.05))
        ax1.set_ylim((-0.05, 1.05))
        ax1.set_title(' '+title[i][:-3],fontsize=size, loc='left',y=0.85)
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.plot([0,1],[0,1], color='black',linestyle=':',linewidth=5) 
        ax1.plot([0,1],[0,p_list[i][0]], color='r',linestyle='dashed',linewidth=5) 
       
        #cbar = ax1.collections[0].colorbar
        cbar = plt.colorbar(im)
        #cbar.ax.tick_params(labelsize=24)
        cbar.set_ticks([])
        cbar.set_label("$k^w_{\min}$                 $k^w_{\max}$",fontsize=size)
        #cbar.ax.set_title('$TC_i$',fontsize=24)
        #cbar.set_label('$C_i$',fontsize=24)

    else:
        ax1=plt.subplot2grid((60,96),((i//4)*21+4,(i%4)*21),colspan=20,rowspan=20)
        im=ax1.scatter(C_list[i],TCi[i],c=kwi[i],s=np.array(ki[i])/max(ki[i])*300,cmap="jet",alpha=0.5)
        ax1.set_xlim((-0.05, 1.05))
        ax1.set_ylim((-0.05, 1.05))
        ax1.set_title(' '+title[i][:-3],fontsize=size, loc='left',y=0.85)
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.plot([0,1],[0,1], color='black',linestyle=':',linewidth=5) 
        ax1.plot([0,1],[0,p_list[i][0]], color='r',linestyle='dashed',linewidth=5) 

    if (i==0) or (i==4):
        ax1.set_ylabel("$TC$",fontsize=size)
        ax1.set_yticks([0.0,0.5,1.0])
        ax1.tick_params(labelsize=size)
    if i>=4:
        ax1.set_xlabel("$C$",fontsize=size)
        ax1.set_xticks([0.0,0.5,1.0])
        ax1.tick_params(labelsize=size)
plt.savefig("TC-C modes.pdf",format='pdf')
# %% 平均TC/C 与 k 的关系 分两段
size=24
fig=plt.figure(figsize=(24, 12))
ax=[]
for i in range(8):
    ax.append(plt.subplot2grid((30,48),((i//4)*12+4,(i%4)*11),colspan=10,rowspan=10))
    cla=(ki[i]>=ki[i].max()/2).astype(int)
    df=pd.DataFrame()
    df['cla']=list(cla)+list(cla)
    df['TCi/Ci']=list(TCi[i])+list(C_list[i])
    n=len(cla)
    hue=['$TC$']*n+['$C$']*n
    df['hue']=hue
    
    ax[i]=sns.boxplot(data=df, x="cla", y="TCi/Ci", hue="hue",width=0.6)

    ax[i].set_xticks([])
    ax[i].set_yticks([])
    ax[i].legend(fontsize=size-8)
    if i !=0:
        ax[i].legend().remove() 
    ax[i].set_title(' '+title[i][:-3],fontsize=size)
    ax[i].tick_params(labelsize=size)
    ax[i].set_ylim((-0.05, 1.05))
    ax[i].set_ylabel('',fontsize=size)
    ax[i].set_xlabel('',fontsize=size)

    if (i==0) or (i==4):
        ax[i].set_yticks([0.0,0.5,1.0])
        # ax[i].set_ylabel('Average $TC$ / $C$',fontsize=size)
    if i>=4:
        ax[i].set_xticks([0,1])
        ax[i].set_xticklabels(['Small','Large'],rotation=0,fontsize=size)
plt.savefig("TC-k.svg",dpi=50,format='svg')
# %%
df

# # %% 平均TC/C 与 k 的关系 分5段
# size=24
# fig=plt.figure(figsize=(24, 12))
# ax=[]
# for i in range(8):
#     ax.append(plt.subplot2grid((30,48),((i//4)*12+4,(i%4)*12),colspan=10,rowspan=10))
#     pct99=np.percentile(ki[i],99)
#     pct95=np.percentile(ki[i],95)
#     pct80=np.percentile(ki[i],80)
#     pct50=np.percentile(ki[i],50)
#     cla=(ki[i]>=pct99).astype(int)+(ki[i]>pct95).astype(int)+(ki[i]>pct80).astype(int)+(ki[i]>pct50).astype(int)
#     ax[i]=sns.pointplot(x=cla,y=TCi[i],color='red',label='$TC_i$')
#     ax[i]=sns.pointplot(x=cla,y=C_list[i],color='blue',label='$C_i$')
#     ax[i].set_xticks([])
#     ax[i].set_yticks([])
#     ax[i].legend(fontsize=size-8)
#     ax[i].set_title(' '+title[i][:-3],fontsize=size, loc='left',y=0.85)
#     ax[i].tick_params(labelsize=size)
#     ax[i].set_xlim((-0.05, 4.05))
#     ax[i].set_ylim((0, 1))

#     if (i==0) or (i==4):
#         ax[i].set_yticks([0.0,0.5,1.0])
#         ax[i].set_ylabel('Average $TC_i$ / $C_i$',fontsize=size)
#     if i>=4:
#         ax[i].set_xlabel("$k_i$ at different %",fontsize=size)
#         ax[i].set_xticks([0,1,2,3,4])
#         ax[i].set_xticklabels(['0%-50%','50%-80%','80%-95%','95%-99%','99%-100%'],rotation=45,fontsize=size)


# %% Ci_kmin
fig=plt.figure(figsize=(24, 12))
data=Ci_kmin
size=24
ax=[]
for i in range(8):
    ax1=plt.subplot2grid((60,96),((i//4)*23+8,(i%4)*21),colspan=20,rowspan=20)
    TCi_dt=np.array(data[i])
    x_list=np.linspace((TCi_dt.shape[0]+1)*1/30,(TCi_dt.shape[0]+1)*9/10-ki0[i].min()+1,4).astype(int)
    x_list_label=np.linspace(ki0[i].min()+(TCi_dt.shape[0]+1)*1/30,(TCi_dt.shape[0]+1)*9/10,4).astype(int)
    a=list(np.vstack((np.array([ki0[i]]),TCi_dt)).T)
    a.sort(key=lambda x:x[0],reverse=True)
    TCi_dt_sort=np.array(a)[:,1:]
    TCi_dt_sort=TCi_dt_sort[:,max(ki0[i].min()-1,1):]
    ax1=sns.heatmap(data=TCi_dt_sort,cmap='jet',mask=TCi_dt_sort<0,cbar=False,yticklabels=[],vmin=0,vmax=1)
    ax1.set_title(title[i][:-3]+' ',fontsize=size,loc='right',y=0.05)
    ax1.set_xticks(x_list)
    ax1.set_xticklabels(x_list_label,ha='center',fontsize=size,rotation=0)
    ax1.set_facecolor("gainsboro")
    if (i==0) or (i==4):
        ax1.set_ylabel('$k_{\min}$                 $k_{\max}$',fontsize=size)
    # if i>=4:
    #     ax1.set_xlabel(xlabel,fontsize=size)
    ax.append(ax1)

position=fig.add_axes([0.81, 0.25,0.012 ,0.45])
cbar=fig.colorbar(ax1.collections[0].colorbar,cax=position,shrink=0.8,cmap='jet')
cbar.ax.tick_params(labelsize=size)
cbar.set_ticks([0.00,0.25,0.50,0.75,1.00])
cbar.ax.set_title('$C$',fontsize=size,x=2.5,y=1.08)

plt.savefig("influence of kmin on C.pdf",format='pdf',dpi=50)

# %% TCi_kmin
fig=plt.figure(figsize=(24, 12))
data=TCi_kmin
size=24
ax=[]
for i in range(8):
    ax1=plt.subplot2grid((60,96),((i//4)*23+8,(i%4)*21),colspan=20,rowspan=20)
    TCi_dt=np.array(data[i])
    x_list=np.linspace((TCi_dt.shape[0]+1)*1/30,(TCi_dt.shape[0]+1)*9/10-ki0[i].min()+1,4).astype(int)
    x_list_label=np.linspace(ki0[i].min()+(TCi_dt.shape[0]+1)*1/30,(TCi_dt.shape[0]+1)*9/10,4).astype(int)
    a=list(np.vstack((np.array([ki0[i]]),TCi_dt)).T)
    a.sort(key=lambda x:x[0],reverse=True)
    TCi_dt_sort=np.array(a)[:,1:]
    TCi_dt_sort=TCi_dt_sort[:,max(ki0[i].min()-1,1):]
    ax1=sns.heatmap(data=TCi_dt_sort,cmap='jet',mask=TCi_dt_sort<0,cbar=False,yticklabels=[],vmin=0,vmax=1)
    ax1.set_title(title[i][:-3]+' ',fontsize=size,loc='right',y=0.05)
    ax1.set_xticks(x_list)
    ax1.set_xticklabels(x_list_label,ha='center',fontsize=size,rotation=0)
    ax1.set_facecolor("gainsboro")
    if (i==0) or (i==4):
        ax1.set_ylabel('$k_{\min}$                 $k_{\max}$',fontsize=size)
    # if i>=4:
    #     ax1.set_xlabel(xlabel,fontsize=size)
    ax.append(ax1)

position=fig.add_axes([0.81, 0.25,0.012 ,0.45])
cbar=fig.colorbar(ax1.collections[0].colorbar,cax=position,shrink=0.8,cmap='jet')
cbar.ax.tick_params(labelsize=size)
cbar.set_ticks([0.00,0.25,0.50,0.75,1.00])
cbar.ax.set_title('$TC$',fontsize=size,x=2.5,y=1.08)
plt.savefig("influence of kmin on TC.pdf",format='pdf',dpi=50)

# %%  Ci_wmin
fig=plt.figure(figsize=(24, 12))
data=Ci_wmin
size=24
ax=[]
for i in range(8):
    ax1=plt.subplot2grid((60,96),((i//4)*23+8,(i%4)*21),colspan=20,rowspan=20)
    TCi_dt=np.array(data[i])
    x_list=np.linspace(1,(TCi_dt.shape[0]+1)*9/10,4).astype(int)
    a=list(np.vstack((np.array([ki0[i]]),TCi_dt)).T)
    a.sort(key=lambda x:x[0],reverse=True)
    TCi_dt_sort=np.array(a)[:,1:]
    ax1=sns.heatmap(data=TCi_dt_sort,cmap='jet',mask=TCi_dt_sort<0,cbar=False,yticklabels=[],vmin=0,vmax=1)
    ax1.set_title(title[i][:-3]+' ',fontsize=size,loc='right',y=0.05,color='white')
    ax1.set_xticks(x_list)
    ax1.set_xticklabels(x_list,ha='center',fontsize=size,rotation=0)
    ax1.set_facecolor("lightsteelblue")
    if (i==0) or (i==4):
        ax1.set_ylabel('$k_{\min}$                 $k_{\max}$',fontsize=size)
    # if i>=4:
    #     ax1.set_xlabel(xlabel,fontsize=size)
    ax.append(ax1)

position=fig.add_axes([0.81, 0.25,0.012 ,0.45])
cbar=fig.colorbar(ax1.collections[0].colorbar,cax=position,shrink=0.8,cmap='jet')
cbar.ax.tick_params(labelsize=size)
cbar.set_ticks([0.00,0.25,0.50,0.75,1.00])
cbar.ax.set_title('$C$',fontsize=size,x=2.5,y=1.08)
plt.savefig("influence of wmin on C.pdf",format='pdf',dpi=50)

# %%  TCi_wmin
fig=plt.figure(figsize=(24, 12))
data=TCi_wmin
size=24
ax=[]
for i in range(8):
    ax1=plt.subplot2grid((60,96),((i//4)*23+8,(i%4)*21),colspan=20,rowspan=20)
    TCi_dt=np.array(data[i])
    x_list=np.linspace(1,(TCi_dt.shape[0]+1)*9/10,4).astype(int)
    a=list(np.vstack((np.array([ki0[i]]),TCi_dt)).T)
    a.sort(key=lambda x:x[0],reverse=True)
    TCi_dt_sort=np.array(a)[:,1:]
    ax1=sns.heatmap(data=TCi_dt_sort,cmap='jet',mask=TCi_dt_sort<0,cbar=False,yticklabels=[],vmin=0,vmax=1)
    ax1.set_title(title[i][:-3]+' ',fontsize=size,loc='right',y=0.05,color='white')
    ax1.set_xticks(x_list)
    ax1.set_xticklabels(x_list,ha='center',fontsize=size,rotation=0)
    ax1.set_facecolor("lightsteelblue")
    if (i==0) or (i==4):
        ax1.set_ylabel('$k_{\min}$                 $k_{\max}$',fontsize=size)
    # if i>=4:
    #     ax1.set_xlabel(xlabel,fontsize=size)
    ax.append(ax1)

position=fig.add_axes([0.81, 0.25,0.012 ,0.45])
cbar=fig.colorbar(ax1.collections[0].colorbar,cax=position,shrink=0.8,cmap='jet')
cbar.ax.tick_params(labelsize=size)
cbar.set_ticks([0.00,0.25,0.50,0.75,1.00])
cbar.ax.set_title('$TC$',fontsize=size,x=2.5,y=1.08)
plt.savefig("influence of wmin on TC.pdf",format='pdf',dpi=50)

# %%
        

