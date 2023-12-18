from env2 import Env
from pathlib import Path
from copy import deepcopy
import numpy as np
from tqdm import tqdm
from env2 import EDGE_NODE_NUM,EDGE_CLUSTER_NUM,CLOUD_NODE_NUM,CLOUD_CLUSTER_NUM,HISTORY_NUM,TASK_PER_CLUSTER,RES_MAP,FPS_MAP,ECOST,CCOST,ECCOST,FAILC,EDGE_BW,CLOUD_BW,EDGE_RTT,CLOUD_RTT,EDGE_MEM,CLOUD_MEM,EDGE_GPU_MEM,CLOUD_GPU_MEM
from scheduler import rong_schedule
import pickle


def eval_rong(eval_config_file="eval_config.txt"):
    env=Env("profile_table-val.db",is_train=False)
    with open(eval_config_file) as f:
        s=f.read().strip().split("\n")
        n=int(s[0])
        tasks=[list(map(int,it.split())) for it in s[1:]]
    tot_reward=0
    anss=[]
    tot_tasks=0
    for tasknum,clusternum in tqdm(tasks):
        states=[[],[],[]]
        for i in range(clusternum):
            t=env.get_state(i)
            # print(t[0])
            res=np.array(t[0],dtype=np.float32)
            taskid=list(t[1].keys())
            pref=np.array([t[1][j] for j in taskid])
            states[0].append(res)
            states[1].append(taskid)
            states[2].append(pref)
        action_dict=dict()
        for i in range(clusternum):
            t=rong_schedule(states[0][i][:tasknum],states[1][i][:tasknum],states[2][i][:tasknum])
            for k,v in t.items():
                action_dict[k]=v

        
        reward,ans=env.submit_action(action_dict,tasknum,clusternum)
        # print(np.mean(reward)-sum(it[3] for it in ans)/len(ans))
        # assert np.abs(np.mean(reward)-sum(it[3] for it in ans)/len(ans))<1e-9
        tot_reward+=np.mean(reward)
        tot_tasks+=tasknum
        anss+=ans
    return sum(it[3] for it in anss)/len(anss),anss

if __name__=="__main__":
    t,anss=eval_rong()
    print(t)
    # print(anss)
    # acc=sorted([it[0] for it in anss])
    # lat=sorted([it[1] for it in anss])
    # cost=sorted([it[2] for it in anss])
    # u=sorted([it[3] for it in anss])
    pickle.dump(anss,open("result_rong.pkl","wb"))