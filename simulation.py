# %%
import numpy as np
import random
import math
import networkx as nx
from itertools import combinations
import matplotlib.pyplot as plt
# 规定两个交互区域
# 中心
A=[0.5,0.5]
B=[-0.5,-0.5]
#半径
r=0.2

# 定义节点数量和迭代次数
num_nodes = 100
T=100
# 初始化节点位置和速度
v=1 #最大速度

nodes = []
for id in range(num_nodes):
    x = random.uniform(-1, 1)# 定义网格尺寸
    y = random.uniform(-1, 1)
    speed = random.uniform(0, v)
    direction = random.uniform(0, 2 * math.pi)
    nodes.append((id, x, y, speed, direction))


#时间的迭代 

Gt_list=[]
Ct_list=[]
Gs=nx.Graph()
edge_num=list()
for t in range(T):
    #在交互区域中的节点
    A_list=[]
    B_list=[]

    # 模拟每个节点的移动
    for node in nodes:
        id, x, y, speed, direction = node
        speed = random.uniform(0, v)
        direction = random.uniform(0, 2 * math.pi)

        # 计算新的位置
        new_x = x + speed * math.cos(direction)
        new_y = y + speed * math.sin(direction)
        
        # 检查是否撞墙
        while(1):
            if new_x < -1 or new_x > 1:
                direction = math.pi - direction
                new_x = x + speed * math.cos(direction)
                new_y = y + speed * math.sin(direction)
            elif new_y < -1 or new_y > 1:
                direction = -direction
                new_x = x + speed * math.cos(direction)
                new_y = y + speed * math.sin(direction)
            else:
                break
        
        # 更新节点位置
        node = (id, new_x, new_y, speed, direction)
        
        # 判断是否在交互区域中
        if (new_x-A[0])**2+(new_y-A[1])**2<=r**2:
            A_list.append(id)
        elif (new_x-B[0])**2+(new_y-B[1])**2<=r**2:
            B_list.append(id)

    #构造timestamp
    # 当k个节点落在交互区域中，两两之间以Cnk的概率发生交互
    # (必然交互)

    Gt = nx.Graph() 
    Gt.add_nodes_from(range(num_nodes))
    Aedges=list(combinations(A_list, 2))
    Gt.add_edges_from(Aedges)
    Bedges=list(combinations(B_list, 2))
    Gt.add_edges_from(Bedges)
    Gt_list.append(Gt)
    edge_num.append(len(Gt.edges()))
    # 计算每一个snapshot的聚类系数Ct
    Ct=nx.clustering(Gt)
    Ct_list.append(Ct)
    #合并到静态网络
    Gs=nx.compose(Gs,Gt)

    #print(Gs.number_of_edges())
# 做T步，生成总网络，计算C
C=nx.clustering(Gs)
#计算p
p=np.array(edge_num).mean()/len(Gs.edges())
# %%
# 计算TC
TC_ti=[]
for t in range(T):
    TC_ti.append(np.array(list(Ct_list[t].values())))
TC_ti=np.array(TC_ti)
#计算每个节点的活跃时间 即度>=1的时间
activeT=np.zeros(num_nodes)
kwi=np.zeros(num_nodes)
for t in range(T):
    ki=np.array(list(Gt_list[t].degree()))[:,1]
    activeT[np.where(ki>0)]=activeT[np.where(ki>0)]+1
    kwi=ki+kwi
# 对活跃时间求平均 得到每个节点的TCi
TCi=[]
for i in range(num_nodes):
    #TCi.append(sum(TC_ti[:,i])/T)
    TCidiv=np.divide(sum(TC_ti[:,i]),activeT[i],np.zeros_like(sum(TC_ti[:,i])),where=activeT[i]!=0)
    TCi.append(TCidiv) #除以活跃时间
TC=np.array(TCi)
C=np.array(list(C.values()))
ki=np.array(list(Gs.degree()))[:,1]

# %%
fig=plt.figure(figsize=(10,7))
size=25
ax1=plt.subplot(1,1,1)
im=ax1.scatter(C,TC,c=kwi+1,s=np.array(ki+1)/max(ki+1)*1000,cmap="jet",alpha=0.5)
ax1.set_xlim((-0.05, 1.05))
ax1.set_ylim((-0.05, 1.05))
title='Simulation'
plt.title(title,size=size+10)
ax1.set_xticks([])
ax1.set_yticks([])
ax1.set_ylabel("$TC$",fontsize=size)
ax1.set_yticks([0.0,0.5,1.0])
ax1.tick_params(labelsize=size)
ax1.set_xlabel("$C$",fontsize=size)
ax1.set_xticks([0.0,0.5,1.0])
ax1.tick_params(labelsize=size)
cbar = plt.colorbar(im)
#cbar.ax.tick_params(labelsize=24)
cbar.set_ticks([])
cbar.set_label("$k^w_{\min}$                                       $k^w_{\max}$",fontsize=size)

ax1.plot([0,1],[0,1], color='black',linestyle=':',linewidth=5) 

ax1.plot([0,1],[0,p], color='r',linestyle='dashed',linewidth=5) 

fig.savefig("simulation.pdf",format="pdf")
