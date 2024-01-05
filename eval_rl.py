from rlmodel import ActorCritic
from env2 import Env
import torch
from pathlib import Path
from copy import deepcopy
import numpy as np
from tqdm import tqdm
from env2 import EDGE_NODE_NUM,EDGE_CLUSTER_NUM,CLOUD_NODE_NUM,CLOUD_CLUSTER_NUM,HISTORY_NUM,TASK_PER_CLUSTER,RES_MAP,FPS_MAP,ECOST,CCOST,ECCOST,FAILC,EDGE_BW,CLOUD_BW,EDGE_RTT,CLOUD_RTT,EDGE_MEM,CLOUD_MEM,EDGE_GPU_MEM,CLOUD_GPU_MEM
import random
import pickle


def load_best_actor_model(actor:ActorCritic):
    l=list(Path("saved_models").glob("actor_*.pth"))
    nums=list(map(lambda x:int(x.stem[6:]),l))
    if len(nums)==0:
        return 0
    else:
        actor.load_state_dict(torch.load(f"saved_models/actor_{max(nums)}.pth"))
        return max(nums)



def eval_rl(actor:ActorCritic,eval_config_file="eval_config.txt",hide_bar=False):
    env=Env("profile_table-val.db",is_train=False)
    with open(eval_config_file) as f:
        s=f.read().strip().split("\n")
        n=int(s[0])
        tasks=[list(map(int,it.split())) for it in s[1:]]
    tot_reward=0
    anss=[]
    tot_tasks=0
    with torch.no_grad():
        reward=torch.tensor([0 for i in range(EDGE_CLUSTER_NUM)],device="cuda")
        for tasknum,clusternum in tqdm(tasks,disable=hide_bar):
            states=[[],[],[],[],[]]
            for i in range(clusternum):
                t=env.get_state(i)
                # print(t[0])
                res=torch.tensor(t[0],device="cuda",dtype=torch.float)
                # tasknum_vec=torch.zeros([TASK_PER_CLUSTER],device="cuda",dtype=torch.float)
                # tasknum_vec[tasknum-1]=1
                taskid=list(t[1].keys())[:tasknum]
                pref=torch.tensor(np.array([t[1][j] for j in taskid]),device="cuda")
                # print("i=",i,"taskid=",taskid)
                history=torch.tensor(t[2],device="cuda",dtype=torch.float)
                states[0].append(res)
                states[1].append(taskid)
                states[2].append(pref)
                states[3].append(history)
                states[4].append(tasknum)
                

            action_dict=dict()
            t=actor(clusternum,torch.tensor(states[4]),torch.stack(states[2]),torch.stack(states[3]))
            for i in range(clusternum):
                # assert len(states[1][i])==tasknum
                for j in range(len(states[1][i])):
                    # print("i=",i,"j=",j,states[1][i][j],tuple(np.argmax(t[k][0][j].cpu().numpy()) for k in range(6)))
                    action_dict[states[1][i][j]]=tuple(np.argmax(t[k][i][j].cpu().numpy()) for k in range(6))

            reward,ans=env.submit_action(action_dict,tasknum,clusternum)
            tot_reward+=np.mean(reward)
            tot_tasks+=clusternum
            anss+=ans
    return sum(it[3] for it in anss)/len(anss),anss
            
if __name__=="__main__":
    actor=ActorCritic().eval().cuda()
    Path("saved_models").mkdir(exist_ok=True)
    last=load_best_actor_model(actor)
        
    t,anss=eval_rl(actor)
    print(t)
    # print(anss)
    # acc=sorted([it[0] for it in anss])
    # lat=sorted([it[1] for it in anss])
    # cost=sorted([it[2] for it in anss])
    # u=sorted([it[3] for it in anss])
    pickle.dump(anss,open("result_rl.pkl","wb"))


