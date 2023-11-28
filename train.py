from rlmodel import Actor,Critic
from env2 import Env
import torch
from pathlib import Path
from copy import deepcopy
import numpy as np
from torch import optim
from tqdm import tqdm
from env2 import EDGE_NODE_NUM,EDGE_CLUSTER_NUM,CLOUD_NODE_NUM,CLOUD_CLUSTER_NUM,HISTORY_NUM,TASK_PER_CLUSTER,RES_MAP,FPS_MAP,ECOST,CCOST,ECCOST,FAILC,EDGE_BW,CLOUD_BW,EDGE_RTT,CLOUD_RTT,EDGE_MEM,CLOUD_MEM,EDGE_GPU_MEM,CLOUD_GPU_MEM
import random


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


def train(batch_size=8,gamma=0.1):
    env=Env("profile_table.db")
    actor=Actor().cuda()
    critic=Critic().cuda()
    Path("saved_models").mkdir(exist_ok=True)
    last=load_best_actor_model(actor)
    load_best_critic_model(critic)
    replay_buf=[]
    actor_optim=optim.Adam(actor.parameters(),lr=1e-3)
    critic_optim=optim.Adam(critic.parameters(),lr=1e-3)
    reward=torch.tensor([0 for i in range(EDGE_CLUSTER_NUM)],device="cuda")
    best_reward=0
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
        
        
        reward=torch.tensor(env.submit_action(action_dict)[0],device="cuda")
        # reward_history=0.9*reward_history+0.1*reward
        new_states=[[],[],[]]
        for i in range(EDGE_CLUSTER_NUM):
            t=env.get_state(i)
            res=torch.tensor(t[0],device="cuda",dtype=torch.float)
            taskid=list(t[1].keys())
            pref=torch.tensor([t[1][j] for j in taskid],device="cuda")
            new_states[0].append(res)
            new_states[1].append(taskid)
            new_states[2].append(pref)

        replay_buf.append((
            torch.stack(states[0]),
            torch.stack(states[2]),
            actions[0],
            actions[1],
            actions[2],
            actions[3],
            actions[4],
            actions[5],
            reward,
            torch.stack(new_states[0]),
            torch.stack(new_states[2])
        ))
        
        if len(replay_buf[0])>1000:
            replay_buf=replay_buf[len(replay_buf)-1000:]
        sample_idx=np.random.choice(np.arange(len(replay_buf)),batch_size)
        loss=0
        aloss=0
        # print(replay_buf[0][7].shape)
        inputs=[torch.stack([replay_buf[i][j]for i in sample_idx]) for j in range(11)]
        # print(inputs[7].shape)
        # new_actions=[[] for _ in range(6)]
        y=0
        new_actions=[[] for _ in range(6)]
        for i in range(EDGE_CLUSTER_NUM):
            t=actor(inputs[9][:,i],inputs[10][:,i])
            for j in range(6):
                new_actions[j].append(t[j])
        for j in range(6):
            new_actions[j]=torch.stack(new_actions[j],dim=1)
        y=critic(inputs[9],inputs[10],new_actions[0],new_actions[1],new_actions[2],new_actions[3],new_actions[4],new_actions[5])
            
        y=inputs[8]+gamma*y
        new_actions=[[] for _ in range(6)]
        for i in range(EDGE_CLUSTER_NUM):
            t=actor(inputs[0][:,i],inputs[1][:,i])
            for j in range(6):
                new_actions[j].append(t[j])
        for j in range(6):
            new_actions[j]=torch.stack(new_actions[j],dim=1)
        q=critic(inputs[0],inputs[1],new_actions[0],new_actions[1],new_actions[2],new_actions[3],new_actions[4],new_actions[5])
        loss+=torch.mean((y-q)**2)
        aloss-=torch.mean(q)
        
        # aloss.backward(retain_graph=True)
        loss.backward()
        # actor_optim.step()
        # actor_optim.zero_grad()
        critic_optim.step()
        critic_optim.zero_grad()
        bar.set_description(f"reward={reward.cpu().numpy()} aloss={aloss:.4f} loss={loss:.4f}")
        if torch.mean(reward)>best_reward:
            best_reward=torch.mean(reward)
            torch.save(actor.state_dict(),f"saved_models/actor_{epoch}.pth")
            torch.save(critic.state_dict(),f"saved_models/critic_{epoch}.pth")
            
if __name__=="__main__":
    train()


