from env2 import Env
from pathlib import Path
from copy import deepcopy
import numpy as np
from tqdm import tqdm
from env2 import EDGE_NODE_NUM,EDGE_CLUSTER_NUM,CLOUD_NODE_NUM,CLOUD_CLUSTER_NUM,HISTORY_NUM,TASK_PER_CLUSTER,RES_MAP,FPS_MAP,ECOST,CCOST,ECCOST,FAILC,EDGE_BW,CLOUD_BW,EDGE_RTT,CLOUD_RTT,EDGE_MEM,CLOUD_MEM,EDGE_GPU_MEM,CLOUD_GPU_MEM,DELAY_THR
import pickle
import sqlite3
import random


def calc_utility_jcab(pref,acc_table,actions,q):
    used_edge_res=[[[0,0,0,0] for i in range(EDGE_NODE_NUM)] for j in range(EDGE_CLUSTER_NUM)]
    used_cloud_res=[[[0,0,0,0] for i in range(CLOUD_NODE_NUM)] for j in range(CLOUD_CLUSTER_NUM)]
    acc=dict()
    latency=dict()
    cost=dict()
    size_client_edge=dict()
    size_edge_cloud=dict()
    now_para=pref
    for it in actions:
        cluster=0
        now_para[it][1]=q[actions[it][3]]
        res=acc_table[(actions[it][0],actions[it][1])]
        acc[it]=res[0]
        sizes=res[1:6]
        edge_res=sum(res[6:6+actions[it][2]]),sum(res[11:11+actions[it][2]])
        cloud_res=sum(res[6+actions[it][2]:11]),sum(res[11+actions[it][2]:16])
        used_edge_res[cluster][actions[it][3]][0]+=1
        used_edge_res[cluster][actions[it][3]][1]+=edge_res[0]
        used_edge_res[cluster][actions[it][3]][2]+=edge_res[1]
        used_cloud_res[actions[it][4]][actions[it][5]][0]+=1
        used_cloud_res[actions[it][4]][actions[it][5]][1]+=cloud_res[0]
        used_cloud_res[actions[it][4]][actions[it][5]][2]+=cloud_res[1]
        size_client_edge[it]=sizes[0]
        size_edge_cloud[it]=sizes[actions[it][2]] if actions[it][2]<5 else 0
        # assert size_client_edge[it]>=0
        # assert env.now_para[it][1]>=0
        used_edge_res[cluster][actions[it][3]][3]+=np.sqrt(sizes[0])
        used_cloud_res[actions[it][4]][actions[it][5]][3]+=np.sqrt(size_edge_cloud[it])
    tot_ans=0
    all_ans=[]
    new_q={it:0 for it in q}
    for it in actions:
        cluster=0
        res=acc_table[actions[it][0],actions[it][1]]
        times=res[16:21]
        edge_calc_lat=sum(times[:actions[it][2]])*used_edge_res[cluster][actions[it][3]][0]
        cloud_calc_lat=sum(times[actions[it][2]:])*used_cloud_res[actions[it][4]][actions[it][5]][0]
        # print(it,now_para[it][1])
        # assert sizes[0]>=0
        # assert now_para[it][1]>=0
        edge_trans_lat=used_edge_res[cluster][actions[it][3]][3]/EDGE_BW/np.sqrt(sizes[0])
        if size_edge_cloud[it]>0:
            cloud_trans_lat=used_cloud_res[actions[it][4]][actions[it][5]][3]/CLOUD_BW/np.sqrt(size_edge_cloud[it])
        else:
            cloud_trans_lat=0
        cost=ECOST*edge_calc_lat+CCOST[actions[it][4]]*cloud_calc_lat+ECCOST*size_edge_cloud[it]
        lat=now_para[it][1]*(edge_calc_lat+cloud_calc_lat)+edge_trans_lat+cloud_trans_lat
        real_lat=edge_calc_lat+cloud_calc_lat+(edge_trans_lat+cloud_trans_lat)
        # print(env.now_para[it][0]*acc[it],lat,env.now_para[it][2]*cost)
        if used_edge_res[cluster][actions[it][3]][1]>EDGE_MEM or used_edge_res[cluster][actions[it][3]][1]>EDGE_GPU_MEM or used_cloud_res[actions[it][4]][actions[it][5]][1]>CLOUD_MEM or used_cloud_res[actions[it][4]][actions[it][5]][1]>CLOUD_GPU_MEM:
            u=FAILC
            new_q[actions[it][3]]+=1
        else:
            u=(now_para[it][0]*acc[it]-now_para[it][2]*cost)*0.1-lat*q[actions[it][3]]
            new_q[actions[it][3]]+=real_lat
        tot_ans+=u/TASK_PER_CLUSTER
        all_ans.append((acc[it],real_lat,cost,u))
    for it in new_q:
        new_q[it]/=sum([actions[jt][3]==it for jt in actions])
        new_q[it]=max(q[it]+new_q[it]-DELAY_THR,0)

    return tot_ans,new_q

def oneslot(pref,acc_table,init_ans,initu,oldq):
    Tmax=10000
    tau=1
    t_no_improve=0
    # print("len=",len(init_ans))
    for _ in range(Tmax):
        u=random.randint(0,len(init_ans)-1)
        new_res=random.randint(0,4)
        new_fr=random.randint(0,4)
        new_step=random.randint(0,5)
        while new_res==init_ans[u][0] and new_fr==init_ans[u][1] and new_step==init_ans[u][2]:
            new_res=random.randint(0,4)
            new_fr=random.randint(0,4)
            new_step=random.randint(0,5)
        new_ans=deepcopy(init_ans)
        new_ans[u][0]=new_res
        new_ans[u][1]=new_fr
        new_ans[u][2]=new_step
        newu,newq=calc_utility_jcab(pref,acc_table,new_ans,oldq)
        # print(initu,newu,newu-initu)
        eta=1/(1+np.exp((initu-newu)/tau))
        if random.random()<=eta:
            init_ans=new_ans
            initu=newu
            oldq=newq
        if newu-initu>=0.001:
            t_no_improve=0
        else:
            t_no_improve+=1
        if t_no_improve>=100:
            break
    # print("iterate num=",_)
    return init_ans,initu,oldq



def ans_init():
    ans=dict()
    q=dict()
    for i in range(TASK_PER_CLUSTER):
        ans[i]=[0,0,5,i%EDGE_NODE_NUM,i%CLOUD_CLUSTER_NUM,(i//CLOUD_CLUSTER_NUM)%CLOUD_NODE_NUM]
    for i in range(EDGE_NODE_NUM):
        q[i]=0
    return ans,q

def profile_jcab():
    con=sqlite3.connect("profile_table.db")
    cur=con.cursor()
    acc=dict()
    for res in range(5):
        for fr in range(5):
            data=cur.execute("select \
                            avg(acc),\
                            avg(size0),\
                            avg(size1),\
                            avg(size2),\
                            avg(size3),\
                            avg(size4),\
                            avg(mem0),\
                            avg(mem1),\
                            avg(mem2),\
                            avg(mem3),\
                            avg(mem4),\
                            avg(gpumem0),\
                            avg(gpumem1),\
                            avg(gpumem2),\
                            avg(gpumem3),\
                            avg(gpumem4),\
                            avg(time0),\
                            avg(time1),\
                            avg(time2),\
                            avg(time3),\
                            avg(time4)\
                            from profile where res=? and fr=?;",(res,fr))
            tablel=data.fetchall()[0]
            acc[(res,fr)]=tablel
    return acc


def eval_jcab():
    acc_table=profile_jcab()
    env=Env("profile_table-val.db",is_train=False)
    tot_reward=0
    anss=[]
    tot_tasks=0
    init_ans=dict()
    q=dict()
    initu=dict()
    for i in range(EDGE_CLUSTER_NUM):
        init_ans[i],q[i]=ans_init()
        initu[i]=0
    while True:
        states=[[],[],[]]
        for i in range(EDGE_CLUSTER_NUM):
            t=env.get_state(i)
            # print(t[0])
            res=np.array(t[0],dtype=np.float32)
            taskid=list(t[1].keys())
            pref=np.array([t[1][j] for j in taskid])
            states[0].append(res)
            states[1].append(taskid)
            states[2].append(pref)
        action_dict=dict()
        for i in range(EDGE_CLUSTER_NUM):
            t,initu[i],q[i]=oneslot(states[2][i],acc_table,init_ans[i],initu[i],q[i])
            init_ans[i]=t
            for k,v in init_ans[i].items():
                action_dict[states[1][i][k]]=v

        
        reward,ans=env.submit_action(action_dict)
        # print(np.mean(reward)-sum(it[3] for it in ans)/len(ans))
        # assert np.abs(np.mean(reward)-sum(it[3] for it in ans)/len(ans))<1e-9
        tot_reward+=np.mean(reward)
        tot_tasks+=EDGE_CLUSTER_NUM
        anss+=ans
        if env.finished():
            break
    return sum(it[3] for it in anss)/len(anss),anss

if __name__=="__main__":
    random.seed(114514)
    t,anss=eval_jcab()
    print(t)
    # print(anss)
    # acc=sorted([it[0] for it in anss])
    # lat=sorted([it[1] for it in anss])
    # cost=sorted([it[2] for it in anss])
    # u=sorted([it[3] for it in anss])
    pickle.dump(anss,open("result_jcab.pkl","wb"))