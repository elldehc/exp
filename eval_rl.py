from rlmodel import Actor,Critic
from env2 import Env
import torch
from pathlib import Path
from copy import deepcopy
import numpy as np
from tqdm import tqdm
from env2 import EDGE_NODE_NUM,EDGE_CLUSTER_NUM,CLOUD_NODE_NUM,CLOUD_CLUSTER_NUM,HISTORY_NUM,TASK_PER_CLUSTER,RES_MAP,FPS_MAP,ECOST,CCOST,ECCOST,FAILC,EDGE_BW,CLOUD_BW,EDGE_RTT,CLOUD_RTT,EDGE_MEM,CLOUD_MEM,EDGE_GPU_MEM,CLOUD_GPU_MEM
import random
import pickle


def load_best_actor_model(actor:Actor):
    l=list(Path("saved_models").glob("actor_*.pth"))
    nums=list(map(lambda x:int(x.stem[6:]),l))
    if len(nums)==0:
        return 0
    else:
        actor.load_state_dict(torch.load(f"saved_models/actor_{max(nums)}.pth"))
        return max(nums)

def load_best_critic_model(critic:Critic):
    l=list(Path("saved_models").glob("critic_*.pth"))
    nums=list(map(lambda x:int(x.stem[7:]),l))
    if len(nums)==0:
        return 0
    else:
        critic.load_state_dict(torch.load(f"saved_models/critic_{max(nums)}.pth"))
        return max(nums)

def random_choice(prob):
    x=random.random()*sum(prob)
    for i in range(len(prob)):
        if x<prob[i]:
            return i
        else:
            x-=prob[i]


def eval_rl(batch_size=8,gamma=0.1):
    env=Env("profile_table.db",is_train=False)
    bar=tqdm(range(100000))
    tot_reward=0
    anss=[]
    tot_tasks=0
    with torch.no_grad():
        actor=Actor().eval().cuda()
        Path("saved_models").mkdir(exist_ok=True)
        last=load_best_actor_model(actor)
        reward=torch.tensor([0 for i in range(EDGE_CLUSTER_NUM)],device="cuda")
        bar=tqdm(range(last+1,100000),desc=f"reward={reward}")
        for epoch in bar:
            states=[[],[],[]]
            for i in range(EDGE_CLUSTER_NUM):
                t=env.get_state(i)
                # print(t[0])
                res=torch.tensor(t[0],device="cuda",dtype=torch.float)
                taskid=list(t[1].keys())
                pref=torch.tensor([t[1][j] for j in taskid],device="cuda")
                states[0].append(res)
                states[1].append(taskid)
                states[2].append(pref)
            actions=[[] for _ in range(6)]
            action_dict=dict()
            for i in range(EDGE_CLUSTER_NUM):
                t=actor(states[0][i],states[2][i])
                for j in range(6):
                    actions[j].append(t[j][0])
                for j in range(t[0].shape[0]):
                    action_dict[states[1][i][j]]=tuple(random_choice(t[k][0][j]) for k in range(6))
            for j in range(6):
                actions[j]=torch.stack(actions[j])
            reward,ans=env.submit_action(action_dict)
            bar.set_description(f"reward={reward}")
            tot_reward+=np.mean(reward)
            tot_tasks+=len(action_dict)
            anss+=ans
            if env.finished():
                break
    return tot_reward/tot_tasks,anss
            
if __name__=="__main__":
    t,anss=eval_rl()
    print(t)
    # print(anss)
    # acc=sorted([it[0] for it in anss])
    # lat=sorted([it[1] for it in anss])
    # cost=sorted([it[2] for it in anss])
    # u=sorted([it[3] for it in anss])
    pickle.dump(anss,open("result_rl.pkl","wb"))


