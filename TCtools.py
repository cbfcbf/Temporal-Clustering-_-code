# %%
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats
from matplotlib import ticker
import json
def init_net(USAL_TN_df):
    # tranform data in series of nx graphs
    iis=np.unique(USAL_TN_df['i'])
    jjs=np.unique(USAL_TN_df['j'])
    nodes=np.union1d(iis,jjs)
    x=np.unique(USAL_TN_df['t'])
    airlines_TN=[]

    for t in range(len(x)):
        fr=USAL_TN_df[USAL_TN_df['t']==x[t]][['i','j']]
        gu=nx.from_pandas_edgelist(fr,'i','j')
        g=nx.Graph()
        g.add_nodes_from(nodes)
        g.add_edges_from(gu.edges)
        g.remove_edges_from(nx.selfloop_edges(g))
        airlines_TN.append(g)

    # Aggregated network

    N=len(nodes)
    AL_AGG=nx.Graph()
    al_agg=np.zeros((N,N))
    for go in airlines_TN:
        AL_AGG=nx.compose(AL_AGG,go)
        al_agg=al_agg+nx.to_numpy_matrix(go)

    G_static=AL_AGG
    G_w = nx.from_numpy_matrix(al_agg)
    #计算数据集性质
    #静态网络G_static
    print("snapshot的数量: ",len(airlines_TN))
    d=dict(G_static.degree())
    print("节点数：", len(G_static.nodes))
    print("静态平均度：", sum(d.values())/len(G_static.nodes))
    print("静态边数：",len(G_static.edges()))
    #动态网络
    edge_num=[]
    for g in airlines_TN:
        edge_num.append(len(g.edges()))
    edge_num=np.array(edge_num)
    k_num=np.array(edge_num)*2/len(G_static.nodes)
    print("动态平均度：", k_num.mean())
    print("动态边数：", edge_num.mean())
    print("动态平均度时序上的方差：", k_num.var())
    p=edge_num.mean()/len(G_static.edges())
    return airlines_TN,G_static,G_w,al_agg,nodes,p
    

def kw_plot(G_w):
    k=list(dict(G_w.degree()).values())
    x=np.array(k)
    num_bin=int(x.max()-x.min())
    hist=plt.hist(x,bins=num_bin,density=True,cumulative=True)
    weight = np.array(list(nx.get_edge_attributes(G_w, "weight").values()))
    w=weight
    num_binw=int(w.max()-w.min())
    histw=plt.hist(w,bins=num_binw,density=True,cumulative=True)
    # 度分布
    fig = plt.figure(figsize=(12,4)) 
    ax1 = plt.subplot2grid((1,5),(0,0),colspan=2,rowspan=1)
    ax2=ax1.twinx()
    ax1.hist(x,bins=num_bin,density=True,color='lightsteelblue',label='histogram')
    ax2.step(hist[1][0:-1],hist[0],color='r',label='histogram-cumulated')
    ax1.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=1))
    ax2.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=1))
    xupb=hist[1][np.where(np.array(hist[0])>0.9)[0][0]]
    ax1.set_xlim(0,xupb)
    ax2.set_ylim(0,1)

    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax1.set_xlabel("k")
    ax1.set_ylabel("percentage",fontsize=16)
    ax2.set_ylabel("cumulative percentage",fontsize=16)
    ax1.set_title("k-distribution",fontsize=25)

    #边权分布
    ax3 =plt.subplot2grid((1,5),(0,3),colspan=2,rowspan=1)
    ax4=ax3.twinx()
    ax3.hist(w,bins=num_binw,density=True,color='lightsteelblue',label='histogram')
    ax4.step(histw[1][0:-1],histw[0],color='r',label='histogram-cumulated')
    ax3.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=1))
    ax4.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=1))
    xupbw=histw[1][np.where(np.array(histw[0])>0.9)[0][0]]
    ax3.set_xlim(0,xupbw)
    ax4.set_ylim(0,1)


    ax3.legend(loc='upper left')
    ax4.legend(loc='upper right')
    ax3.set_xlabel("k")
    ax3.set_ylabel("percentage",fontsize=16)
    ax4.set_ylabel("cumulative percentage",fontsize=16)
    ax3.set_title("w-distribution",fontsize=25)

#相关函数
def G_tdelta_generator(delta,AggG,Gt,nodes):
    k=0
    T=np.shape(Gt)[0]
    G_tdelta=[]
    x=np.array([d[1] for d in AggG.degree()])
    set_k=set(nodes[np.where(x>k)[0]])
    vec_k=nodes[np.where(x>k)[0]]
    size_Sk=len(vec_k)
    if size_Sk>3:
        for t in range(T-delta):
            g=Gt[t]
            neighs=[[] for h in range(size_Sk)] #用来储存子图中第i个节点连接的所有子图节点
            for node in range(size_Sk):
                deh=set(nx.neighbors(Gt[t],vec_k[node])) #某个度数大于k的vec_k[node]节点所有连接的节点
                neighs[node]=np.array(list(set_k & deh)) #与set_k取交集 排除子图之外的边
            for D in range(delta):
                for n in range(size_Sk):
                    doh=set(nx.neighbors(Gt[t+D],vec_k[n]))
                    neighs[n]=np.array(list(set(neighs[n]) & doh)) #再取交集
            
            #利用vec_k 和 neighs生成一系列子图
            edges_dict={}
            for node in range(size_Sk):
                edges_dict[vec_k[node]]=list(neighs[node])

            g_tdelta=nx.from_dict_of_lists(edges_dict)
            g_tdelta.remove_edges_from(nx.selfloop_edges(g))
            G_tdelta.append(g_tdelta)
    return G_tdelta
# 改变时间窗口，生成G_dt 
def G_dt_generator(dt,Gt):#dt几个月作为时间窗口 #滑动
    G_dt=[]
    for i in range(len(Gt)-dt+1):
        G_dt_i=nx.Graph()
        for j in range(dt):
            G_dt_i=nx.compose(G_dt_i,Gt[i+j])
        G_dt.append(G_dt_i)

    return G_dt

def G_dt_generator2(dt,Gt):#dt几个月作为时间窗口 #非滑动
    G_dt=[]
    for i in range(0,len(Gt)-dt+1,dt):
        G_dt_i=nx.Graph()
        for j in range(dt):
            G_dt_i=nx.compose(G_dt_i,Gt[i+j])
        G_dt.append(G_dt_i)
    edge_num=[]
    for g in G_dt:
        edge_num.append(len(g.edges()))
    edge_num=np.array(edge_num)
    k_num=np.array(edge_num)*2/len(g.nodes) 
    # print("snapshot的数量: ",len(G_dt))
    # print("动态平均度：", k_num.mean())
    # print("动态边数：", edge_num.mean())
    # print("动态平均度时序上的方差：", k_num.var())
    
    return G_dt,edge_num.mean()
    # return G_dt

def find_index(G_static,G_tdelta,G_w):
    #计算静态网络聚类系数
    Ci=nx.clustering(G_static)
    C_list=list(Ci.values())
    # 计算此Δ下不同节点不同时间的局部聚类系数矩阵
    TC_ti=[]
    for i in range(len(G_tdelta)):
        TC_t=nx.clustering(G_tdelta[i])
        TC_ti.append(np.array(list(TC_t.values())))
    TC_ti=np.array(TC_ti)
    #计算每个节点的活跃时间 即度>1的时间
    activeT=np.zeros(len(G_tdelta[0].nodes))

    for t in range(len(G_tdelta)):
        ki=np.array(list(G_tdelta[t].degree()))[:,1]
        activeT[np.where(ki>0)]=activeT[np.where(ki>0)]+1

    # 对活跃时间求平均 得到每个节点的TCi
    TCi=[]
    T,I=TC_ti.shape
    for i in range(I):
        #TCi.append(sum(TC_ti[:,i])/T)
        TCidiv=np.divide(sum(TC_ti[:,i]),activeT[i],np.zeros_like(sum(TC_ti[:,i])),where=activeT[i]!=0)
        TCi.append(TCidiv) #除以活跃时间
        #TCi.append(max(TC_ti[:,i]))#替代选项
    ki=[k[1] for k in list(G_static.degree())]
    kwi=[k[1] for k in list(G_w.degree(weight='weight'))]

    return C_list,TCi,ki,kwi,TC_ti

def TCplot(C_list,TCi,ki,kwi,title='US Air Traffic TN'):
    # 核估计
    X=np.array(C_list)
    Y=np.array(TCi)
    values = np.vstack([X,Y])
    kernel = stats.gaussian_kde(values)
    Xd, Yd = np.mgrid[0:1:100j, 0:1:100j]
    positions = np.vstack([Xd.ravel(), Yd.ravel()])
    Z = np.reshape(kernel(positions).T, Xd.shape)
    Z=np.rot90(Z)
    test_data=np.linspace(0,1,100)
    pred=np.sum(Z/np.sum(Z,axis=0)*np.rot90(Yd),axis=0)

    kernelw = stats.gaussian_kde(values,weights=np.array(kwi))
    Zw = np.reshape(kernelw(positions).T, Xd.shape)
    Zw=np.rot90(Zw)
    predw=np.sum(Zw/np.sum(Zw,axis=0)*np.rot90(Yd),axis=0)

    corr=round(np.corrcoef(X,Y)[1,0],4)

    fig=plt.figure(figsize=(12, 10))
    fig.suptitle(title+', corr='+str(corr),fontsize=25)
    ax1=plt.subplot2grid((10,11),(3,0),colspan=6,rowspan=7)
    im=ax1.scatter(C_list,TCi,c=kwi,s=np.array(ki)/max(ki)*300,cmap="jet",alpha=0.5)
    ax1.set_xlabel("$C_i$",fontsize=16)
    ax1.set_ylabel("$TC_i$",fontsize=16)
    ax1.set_xlim((-0.05, 1.05))
    ax1.set_ylim((-0.05, 1.05))
    ax1.plot([0,1],[0,1], 'r:')

    cb=fig.colorbar(im,ax=ax1,label='$k^w_i$',orientation='horizontal')
    cb.set_label('$k^w_i$',fontsize=16)
    ax2=plt.subplot2grid((10,11),(3,6),colspan=1,rowspan=5)
    ax2.hist(TCi,bins=15,
    orientation='horizontal',color="lightsteelblue")
    ax2.set_ylim((-0.05, 1.05))
    ax2.set_yticks([])
    ax3=plt.subplot2grid((10,11),(2,0),colspan=6,rowspan=1)
    ax3.hist(C_list,bins=15,color="lightsteelblue")
    ax3.set_xlim((-0.05, 1.05))
    ax3.set_xticks([])
    ax4=plt.subplot2grid((10,11),(0,0),colspan=6,rowspan=2)
    ax4.set_xlim((-0.05, 1.05))
    ax4.set_xticks([])
    ax4.plot(test_data,pred,color='r')
    ax4.plot(test_data,predw,color='b')
    ax4.legend( labels=["$\mathbb{E}[TC_i|C_i]$, unweighted","$\mathbb{E}[TC_i|C_i]$, weighted"] )
    ax5=plt.subplot2grid((10,11),(0,8),colspan=3,rowspan=3)
    ax5=sns.kdeplot(x=C_list,y=TCi,fill=True,cmap="Blues")
    ax5.set_xlabel("$C_i$",fontsize=16)
    ax5.set_ylabel("$TC_i$",fontsize=16)
    ax5.set_xlim((-0.05, 1.05))
    ax5.set_ylim((-0.05, 1.05))
    ax5.set_title("kde: unweighted",fontsize=16)
    ax6=plt.subplot2grid((10,11),(4,8),colspan=3,rowspan=3)
    ax6=sns.kdeplot(x=C_list,y=TCi,fill=True,cmap="Blues",weights=ki)
    ax6.set_xlabel("$C_i$",fontsize=16)
    ax6.set_ylabel("$TC_i$",fontsize=16)
    ax6.set_xlim((-0.05, 1.05))
    ax6.set_ylim((-0.05, 1.05))
    ax6.set_title("kde: weighted by $k_i$",fontsize=16)
    return corr

def find_C_w(al_agg,w_min=0):

    G_w = nx.from_numpy_matrix(al_agg)
    al_aggpct=al_agg.copy() #防止之后对al_agg改动
    al_aggpct[np.where(al_agg<w_min)]=0
    G_w_pct=nx.from_numpy_matrix(al_aggpct)
    Ci_pct=nx.clustering(G_w_pct)
    C_pct_list=list(Ci_pct.values())

    return C_pct_list

def draw(w_min,k_min,dt,Gt,G_w,G_static,al_agg,title):
    G_dt=G_dt_generator(dt=dt,Gt=Gt)
    C_list,TCi,ki,kwi=find_index(G_static=G_static,G_tdelta=G_dt,G_w=G_w)
    C_list_w=find_C_w(al_agg,w_min=w_min)
    ki=np.array(ki)
    C_list_w=np.array(C_list_w)
    TCi=np.array(TCi)
    C_list_w_k=C_list_w[np.where(ki>k_min)]
    TCi_k=TCi[np.where(ki>k_min)]
    kwi=np.array(kwi)
    ki_k=ki[np.where(ki>k_min)]
    kwi_k=kwi[np.where(ki>k_min)]

    corr=TCplot(C_list_w_k,TCi_k,ki_k,kwi_k,title=title+', $dt='+str(dt)+'$, $w_{min}='+str(w_min)+'$, $k_{min}='+str(k_min)+'$')
    return corr

def TCplot_classs(C_list,TCi,ki,kwi,class_nodes1,class_nodes2,title='US Air Traffic TN',cname1='class1',cname2='class2'):
    # 核估计
    C_list=np.array(C_list)
    TCi=np.array(TCi)
    kwi=np.array(kwi)
    ki=np.array(ki)
    X=np.array(C_list)
    Y=np.array(TCi)
    values = np.vstack([X,Y])
    kernel = stats.gaussian_kde(values)
    Xd, Yd = np.mgrid[0:1:100j, 0:1:100j]
    positions = np.vstack([Xd.ravel(), Yd.ravel()])
    Z = np.reshape(kernel(positions).T, Xd.shape)
    Z=np.rot90(Z)
    test_data=np.linspace(0,1,100)
    pred=np.sum(Z/np.sum(Z,axis=0)*np.rot90(Yd),axis=0)

    kernelw = stats.gaussian_kde(values,weights=np.array(kwi))
    Zw = np.reshape(kernelw(positions).T, Xd.shape)
    Zw=np.rot90(Zw)
    predw=np.sum(Zw/np.sum(Zw,axis=0)*np.rot90(Yd),axis=0)

    corr=round(np.corrcoef(X,Y)[1,0],4)

    fig=plt.figure(figsize=(12, 10))
    fig.suptitle(title+', corr='+str(corr),fontsize=25)
    ax1=plt.subplot2grid((10,11),(3,0),colspan=6,rowspan=5)
    ax1.scatter(C_list[class_nodes1],TCi[class_nodes1],c='r',s=np.array(ki[class_nodes1])/max(ki)*300,alpha=0.5,label=cname1)
    ax1.scatter(C_list[class_nodes2],TCi[class_nodes2],c='b',s=np.array(ki[class_nodes2])/max(ki)*300,alpha=0.5,label=cname2)
    ax1.set_xlabel("$C_i$",fontsize=16)
    ax1.set_ylabel("$TC_i$",fontsize=16)
    ax1.set_xlim((-0.05, 1.05))
    ax1.set_ylim((-0.05, 1.05))
    ax1.legend()

    ax2=plt.subplot2grid((10,11),(3,6),colspan=1,rowspan=5)
    ax2.hist(TCi,bins=15,
    orientation='horizontal',color="lightsteelblue")
    ax2.set_ylim((-0.05, 1.05))
    ax2.set_yticks([])
    ax3=plt.subplot2grid((10,11),(2,0),colspan=6,rowspan=1)
    ax3.hist(C_list,bins=15,color="lightsteelblue")
    ax3.set_xlim((-0.05, 1.05))
    ax3.set_xticks([])
    ax4=plt.subplot2grid((10,11),(0,0),colspan=6,rowspan=2)
    ax4.set_xlim((-0.05, 1.05))
    ax4.set_xticks([])
    ax4.plot(test_data,pred,color='r')
    ax4.plot(test_data,predw,color='b')
    ax4.legend( labels=["$\mathbb{E}[TC_i|C_i]$, unweighted","$\mathbb{E}[TC_i|C_i]$, weighted"] )
    ax5=plt.subplot2grid((10,11),(0,8),colspan=3,rowspan=3)
    ax5=sns.kdeplot(x=C_list,y=TCi,fill=True,cmap="Blues")
    ax5.set_xlabel("$C_i$",fontsize=16)
    ax5.set_ylabel("$TC_i$",fontsize=16)
    ax5.set_xlim((-0.05, 1.05))
    ax5.set_ylim((-0.05, 1.05))
    ax5.set_title("kde: unweighted",fontsize=16)
    ax6=plt.subplot2grid((10,11),(4,8),colspan=3,rowspan=3)
    ax6=sns.kdeplot(x=C_list,y=TCi,fill=True,cmap="Blues",weights=kwi)
    ax6.set_xlabel("$C_i$",fontsize=16)
    ax6.set_ylabel("$TC_i$",fontsize=16)
    ax6.set_xlim((-0.05, 1.05))
    ax6.set_ylim((-0.05, 1.05))
    ax6.set_title("kde: weighted by degree",fontsize=16)
    return corr

def find_index2(G_static,G_t,G_w,k_min,w_min): #先筛选kmin再筛选wmin
    #计算静态网络聚类系数 带筛选
    ki0=[k[1] for k in list(G_w.degree())]
    kwi0=[k[1] for k in list(G_w.degree(weight='weight'))]

    G_w_kmin=G_w.copy()
    G_static_kmin2=G_static.copy()
    node_removed=list(np.where(np.array(ki0)<k_min)[0])
    node_removed2=list(np.array(G_static.nodes())[np.where(np.array(ki0)<k_min)[0]])
    G_w_kmin.remove_nodes_from(node_removed)
    G_static_kmin2.remove_nodes_from(node_removed2)

    weight = np.array(list(nx.get_edge_attributes(G_w_kmin, "weight").values()))
    G_w_kmin_wmin=G_w_kmin.copy()
    edge_removed=np.array(G_w_kmin_wmin.edges())[list(np.where(weight<w_min))[0]]
    edge_removed2=np.array(G_static_kmin2.edges())[list(np.where(weight<w_min))[0]]
    G_w_kmin_wmin.remove_edges_from(edge_removed)

    #? 检查节点附近是否还有度 否则再一次删除节点

    G_w2=G_w_kmin_wmin.copy()
    #print(len(G_w2.nodes()))

    Ci=nx.clustering(G_w2)
    C_list=list(Ci.values())

    C_list_total=np.zeros(len(G_w.nodes()))-1
    C_list_total[list(Ci.keys())]=list(Ci.values())

    # 计算此Δ下不同节点不同时间的局部聚类系数矩阵
    TC_ti=[]
    G_t2=[]
    for t in range(len(G_t)):
        G_t2.append(G_t[t].copy())
        G_t2[t].remove_nodes_from(node_removed2)
        G_t2[t].remove_edges_from(edge_removed2)

        TC_i=nx.clustering(G_t2[t])
        TC_ti.append(np.array(list(TC_i.values())))


    TC_ti=np.array(TC_ti)
    #计算每个节点的活跃时间 即度>=1的时间
    activeT=np.zeros(len(G_t2[0].nodes))
    for t in range(len(G_t2)):
        ki=np.array(list(G_t2[t].degree()))[:,1]
        activeT[np.where(ki>0)]=activeT[np.where(ki>0)]+1

    # 对活跃时间求平均 得到每个节点的TCi
    TCi=[]
    TCi_total=[]
    T,I=TC_ti.shape
    #print(I)
    for i in range(I):
        TCidiv=np.divide(sum(TC_ti[:,i]),activeT[i],np.zeros_like(sum(TC_ti[:,i])).astype(np.float32),where=activeT[i]!=0)
        TCi.append(TCidiv) #除以活跃时间

    TCi_total=np.zeros(len(G_w.nodes()))-1
    TCi_total[list(Ci.keys())]=TCi

    TCi=np.array(TCi)
    TCi_total=np.array(TCi_total)
    ki=[k[1] for k in list(G_w2.degree())]
    kwi=[k[1] for k in list(G_w2.degree(weight='weight'))]
    return C_list,TCi,ki,kwi,C_list_total,TCi_total,kwi0,ki0


def draw_map(data,ki,title,xlabel):
    TCi_dt=np.array(data)
    x_list=range(1,TCi_dt.shape[0]+1,round(TCi_dt.shape[0]/20))
    a=list(np.vstack((np.array([ki]),TCi_dt)).T)
    a.sort(key=lambda x:x[0],reverse=True)
    TCi_dt_sort=np.array(a)[:,1:]
    plt.figure(dpi=120)
    ax=sns.heatmap(data=TCi_dt_sort,cmap='jet',mask=TCi_dt_sort<0,yticklabels=[],xticklabels=round(TCi_dt.shape[0]/20))
    plt.title(title,fontsize=10)
    plt.ylabel('$k_{min}$ → $k_{max}$',fontsize=16)
    plt.xlabel(xlabel,fontsize=16)
    ax.set_xticklabels(x_list,ha='center')
    ax.set_facecolor("lightsteelblue")

def draw2(Gt,G_w,G_static,title,w_min=1,k_min=1,dt=1,flag=1):
    if flag==1:
        G_dt=G_dt_generator(dt=dt,Gt=Gt)
    else:
        G_dt=G_dt_generator2(dt=dt,Gt=Gt)    
    C_list,TCi,ki,kwi,C_list_total,TCi_total,kwi0,ki0=find_index2(G_static=G_static,G_t=G_dt,G_w=G_w,k_min=k_min,w_min=w_min)
    edge_num=[]
    for g in G_dt:
        edge_num.append(len(g.edges()))
    edge_num=np.array(edge_num)
    p=round(edge_num.mean()/len(G_static.edges()),2)
    corr=TCplot(C_list,TCi,ki,kwi,title=title+', $dt='+str(dt)+'$, $w_{min}='+str(w_min)+'$, $k_{min}='+str(k_min)+'$')
    return corr


def calTC(Gt,G_w,G_static,title,w_min=1,k_min=1,dt=1):
    G_dt,ed=G_dt_generator2(dt=dt,Gt=Gt) 
    C_list,TCi,ki,kwi,C_list_total,TCi_total,kwi0,ki0=find_index2(G_static=G_static,G_t=G_dt,G_w=G_w,k_min=k_min,w_min=w_min)
    return C_list,TCi,ki,kwi

def readdata(s):
    ordered_IDs=[]
    if s=="US Air Traffic":
        df=pd.read_csv('../dataset/US Air Traffic TN/USAL_TN_monthly_2012_2020.txt', header=None, sep="\s+")
        title='US Air Traffic TN'
        ordered_IDs = pd.read_csv('../dataset/US Air Traffic TN/ordered_airport_ID.txt', header=None, sep="\s+")
        df.columns=(['t','i','j'])
        w_min=1
        k_min=10
        dt=10
        maxw=50
    elif s=="Hospital":
        df=pd.read_csv("../dataset/Hospital TN/hospital_TN2.txt", sep="\s+")
        title='Hospital TN'
        df.columns=(['ddt','i','j','tagi','tagj','t'])
        w_min=1
        k_min=1
        dt=10
        maxw=30
    elif s=="PhD Exchange":
        df=pd.read_csv("../dataset/PhD Exchange TN/PhD_exchange.txt",sep="\s+")
        title='PhD Exchange TN'
        df.columns=(['i','j','w','t'])
        w_min=1
        k_min=1
        dt=10
        maxw=15
    elif s=="Workplace":
        df=pd.read_csv("../dataset/Workplace TN/workplace_TN.txt", header=None, sep="\s+")
        title='Workplace TN'
        df.columns=(['t','i','j'])
        w_min=1
        k_min=1
        dt=50
        maxw=30
    elif s=="Primary School":
        df=pd.read_csv('../dataset/Primary School TN/Primary_school_5_min_2_DAYS.txt', header=None, sep="\s+")
        title='Primary School TN'
        df.columns=(['t','i','j'])
        w_min=2
        k_min=1
        dt=10
        maxw=50
    elif s=="High School":
        df=pd.read_csv("../dataset/Highschool TN/highschool TN.txt",header=None, sep="\s+")
        title='High School TN'
        df.columns=(['dt','i','j','tagi','tagj','t'])
        w_min=2
        k_min=1
        dt=5
        maxw=30
    elif s=="Village":
        df=pd.read_csv("../dataset/Village TN/village TN.txt", sep="\s+")
        title='Village TN'
        df.columns=(['dt','i','j','t'])
        w_min=1
        k_min=1
        dt=20
        maxw=30
    elif s=="SFHH":
        df=pd.read_csv("../dataset/SFHH TN/SFHH TN.txt", sep="\s+")
        title='SFHH TN'
        df.columns=(['dt','i','j','t'])
        w_min=1
        k_min=1
        dt=1
        maxw=15
    else: 
        return 0,0,0,0,0,0,0

    return df,title,w_min,k_min,dt,maxw,ordered_IDs

class NpEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)
        

def read_draw_data(titles):
    C_list=[]
    TCi=[]
    ki=[]
    kwi=[]
    Ci_kmin=[]
    ki0=[]
    TCi_kmin=[]
    Ci_wmin=[]
    TCi_wmin=[]
    title=[]
    k_min=[]
    w_min=[]
    dt=[]
    maxw=[]
    maxk=[]
    p_list=[]
    # titles=['Hospital TN']
    for title_data in titles:
        with open(title_data+".json") as f_obj:
            df2 = json.load(f_obj)  #读取文件
        C_list.append(np.array(df2["C"]))
        TCi.append(np.array(df2["TC"]))
        ki.append(np.array(df2["ki"]))
        kwi.append(np.array(df2["kwi"]))
        Ci_kmin.append(np.array(df2["Ci_kminT"]).T)
        ki0.append(np.array(df2["ki0"]))
        TCi_kmin.append(np.array(df2["TCi_kminT"]).T)
        Ci_wmin.append(np.array(df2["Ci_wminT"]).T)
        TCi_wmin.append(np.array(df2["TCi_wminT"]).T)

        title.append(df2["title"])
        k_min.append(df2["k_min"])
        w_min.append(df2["w_min"])
        dt.append(df2["dt"])
        maxw.append(df2["maxw"])
        maxk.append(df2["maxk"])
        p_list.append(df2['p'])

    return C_list,TCi,ki,kwi,Ci_kmin,ki0,TCi_kmin,Ci_wmin,TCi_wmin,title,k_min,w_min,dt,maxw,maxk,p_list