from rlmodel import ActorCritic
from env2 import Env
import torch
from pathlib import Path
from copy import deepcopy
import numpy as np
from torch import optim
from tqdm import tqdm
from env2 import EDGE_NODE_NUM,EDGE_CLUSTER_NUM,CLOUD_NODE_NUM,CLOUD_CLUSTER_NUM,HISTORY_NUM,TASK_PER_CLUSTER,RES_MAP,FPS_MAP,ECOST,CCOST,ECCOST,FAILC,EDGE_BW,CLOUD_BW,EDGE_RTT,CLOUD_RTT,EDGE_MEM,CLOUD_MEM,EDGE_GPU_MEM,CLOUD_GPU_MEM
import random
from itertools import chain
from torch.distributions import Categorical
from eval_rl import eval_rl


def load_best_actor_model(actor:ActorCritic):
    l=list(Path("saved_models").glob("actor_*.pth"))
    nums=list(map(lambda x:int(x.stem[6:]),l))
    if len(nums)==0:
        return 0
    else:
        actor.load_state_dict(torch.load(f"saved_models/actor_{max(nums)}.pth"))
        return max(nums)


def train(batch_size=32,gamma=0.5):
    env=Env("profile_table.db")
    actor=ActorCritic().cuda()
    Path("saved_models").mkdir(exist_ok=True)
    last=load_best_actor_model(actor)
    replay_buf=[]
    actor_optim=optim.Adam(chain(actor.parameters()),lr=1e-3)
    # critic_optim=optim.Adam(critic.parameters(),lr=1e-3)
    best_reward=eval_rl(actor,hide_bar=True)[0]
    bar=tqdm(range(last+1,10000000),desc=f"reward={-1} best_reward={best_reward}")
    loss=0
    aloss=0
    for epoch in bar:
        reward=torch.tensor([0 for i in range(EDGE_CLUSTER_NUM)],device="cuda")
        replay_buf=[]
        for step in range(batch_size):
            clusternum=random.randint(1,EDGE_CLUSTER_NUM)
            tasknum=random.randint(1,TASK_PER_CLUSTER)
            states=[[],[],[],[],[]]
            for i in range(clusternum):
                t=env.get_state(i)
                # print(t[0])
                res=torch.tensor(t[0],device="cuda",dtype=torch.float)
                # tasknum_vec=torch.zeros([TASK_PER_CLUSTER],device="cuda",dtype=torch.float)
                # tasknum_vec[tasknum-1]=1
                taskid=list(t[1].keys())[:tasknum]
                # print("i=",i,"taskid=",taskid)
                history=torch.tensor(t[2],device="cuda",dtype=torch.float)
                pref=torch.tensor(np.array([t[1][j] for j in taskid]),device="cuda")
                states[0].append(res)
                states[1].append(taskid)
                states[2].append(pref)
                states[3].append(history)
                states[4].append(tasknum)
            action_dict=dict()
            action_prob=[]
            t=actor(clusternum,torch.tensor(states[4]),torch.stack(states[2]),torch.stack(states[3]))
            for i in range(clusternum):
                t_action_prob=0
                for j in range(tasknum):
                    m = [Categorical(t[k][i][j]) for k in range(6)]
                    choices=[m[k].sample() for k in range(6)]
                    t_action_prob+=(sum([m[k].log_prob(choices[k]) for k in range(6)]))
                    # print("i=",i,"j=",j,states[1][i][j],tuple(choices[k].item() for k in range(6)))
                    action_dict[states[1][i][j]]=tuple(choices[k].item() for k in range(6))
                action_prob.append(t_action_prob/tasknum/clusternum)
            
            
            reward=torch.tensor(env.submit_action(action_dict,tasknum,clusternum)[0],device="cuda")            
            values=t[6]
            action_prob=torch.stack(action_prob)
            # print(reward.shape,values.shape,action_prob.shape)

            replay_buf.append((
                torch.mean(reward),
                torch.mean(values),
                torch.mean(action_prob),
            ))
        
        sample_idx=np.arange(batch_size)
        loss=0
        aloss=0
        # print(replay_buf[0][7].shape)
        inputs=[torch.stack([replay_buf[i][j]for i in sample_idx]) for j in range(3)]
        for i in range(inputs[0].shape[0]-2,-1,-1):
            inputs[0][i]+=inputs[0][i+1]*gamma
        inputs[0]=(inputs[0]-torch.mean(inputs[0]))/(torch.std(inputs[0])+1e-9)
        # new_actions=[[] for _ in range(6)]
        loss+=torch.mean(torch.nn.functional.smooth_l1_loss(inputs[1],inputs[0]))
        # print(inputs[8].shape,inputs[9].shape,inputs[10].shape)
        aloss-=torch.mean((inputs[0]-inputs[1]).detach()*inputs[2])
        
        sloss=loss+0.1*aloss
        sloss.backward()
        # loss.backward()
        actor_optim.step()
        actor_optim.zero_grad()
        # critic_optim.step()
        # critic_optim.zero_grad()
        eval_reward=eval_rl(actor,hide_bar=True)[0]
        if eval_reward>best_reward:
            best_reward=eval_reward
            torch.save(actor.state_dict(),f"saved_models/actor_{epoch}.pth")
        bar.set_description(f"reward={eval_reward:.4f} best_reward={best_reward:.4f} aloss={aloss:.4f} loss={loss:.4f}")
        
            
if __name__=="__main__":
    train()


