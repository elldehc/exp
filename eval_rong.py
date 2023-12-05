from env2 import Env
import torch
from pathlib import Path
from copy import deepcopy
import numpy as np
from tqdm import tqdm
from env2 import EDGE_NODE_NUM,EDGE_CLUSTER_NUM,CLOUD_NODE_NUM,CLOUD_CLUSTER_NUM,HISTORY_NUM,TASK_PER_CLUSTER,RES_MAP,FPS_MAP,ECOST,CCOST,ECCOST,FAILC,EDGE_BW,CLOUD_BW,EDGE_RTT,CLOUD_RTT,EDGE_MEM,CLOUD_MEM,EDGE_GPU_MEM,CLOUD_GPU_MEM
from scheduler import rong_schedule
import pickle


def eval_rong():
    env=Env("profile_table-val.db",is_train=False)
    tot_reward=0
    anss=[]
    tot_tasks=0
    while True:
        states=[[],[],[]]
        for i in range(EDGE_CLUSTER_NUM):
            t=env.get_state(i)
            # print(t[0])
            res=torch.tensor(t[0],dtype=torch.float)
            taskid=list(t[1].keys())
            pref=torch.tensor([t[1][j] for j in taskid])
            states[0].append(res)
            states[1].append(taskid)
            states[2].append(pref)
        action_dict=dict()
        for i in range(EDGE_CLUSTER_NUM):
            t=rong_schedule(states[0][i],states[1][i],states[2][i])
            for k,v in t.items():
                action_dict[k]=v

        
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
    t,anss=eval_rong()
    print(t)
    # print(anss)
    # acc=sorted([it[0] for it in anss])
    # lat=sorted([it[1] for it in anss])
    # cost=sorted([it[2] for it in anss])
    # u=sorted([it[3] for it in anss])
    pickle.dump(anss,open("result_rong.pkl","wb"))