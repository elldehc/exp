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
from itertools import chain
from torch.distributions import Categorical
from eval_rl import eval_rl


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
            return i,prob[i]
        else:
            x-=prob[i]


def train(batch_size=128,gamma=0.9):
    env=Env("profile_table.db")
    actor=Actor().cuda()
    critic=Critic().cuda()
    Path("saved_models").mkdir(exist_ok=True)
    last=load_best_actor_model(actor)
    load_best_critic_model(critic)
    replay_buf=[]
    actor_optim=optim.Adam(chain(actor.parameters(),critic.parameters()),lr=1e-3)
    # critic_optim=optim.Adam(critic.parameters(),lr=1e-3)
    best_reward=eval_rl(actor)[0]
    bar=tqdm(range(last+1,10000000),desc=f"reward={-1} best_reward={best_reward}")
    loss=0
    aloss=0
    for epoch in bar:
        reward=torch.tensor([0 for i in range(EDGE_CLUSTER_NUM)],device="cuda")
        replay_buf=[]
        for step in range(batch_size):
            states=[[],[],[]]
            for i in range(EDGE_CLUSTER_NUM):
                t=env.get_state(i)
                # print(t[0])
                res=torch.tensor(t[0],device="cuda",dtype=torch.float)
                taskid=list(t[1].keys())
                pref=torch.tensor(np.array([t[1][j] for j in taskid]),device="cuda")
                states[0].append(res)
                states[1].append(taskid)
                states[2].append(pref)
            actions=[[] for _ in range(6)]
            action_dict=dict()
            action_prob=[]
            for i in range(EDGE_CLUSTER_NUM):
                cluster=torch.zeros([EDGE_CLUSTER_NUM],device="cuda",dtype=torch.float)
                cluster[i]=1
                t=actor(states[0][i],cluster,states[2][i])
                for j in range(6):
                    actions[j].append(t[j][0])
                for j in range(t[0].shape[0]):
                    m = [Categorical(t[k][0][j]) for k in range(6)]
                    choices=[m[k].sample() for k in range(6)]
                    action_prob.append(sum([m[k].log_prob(choices[k]) for k in range(6)]))
                    action_dict[states[1][i][j]]=tuple(choices[k].item() for k in range(6))
            for j in range(6):
                actions[j]=torch.stack(actions[j])
            
            
            reward=reward*gamma+torch.tensor(env.submit_action(action_dict)[0],device="cuda")
            # reward_history=0.9*reward_history+0.1*reward
            # new_states=[[],[],[]]
            # for i in range(EDGE_CLUSTER_NUM):
            #     t=env.get_state(i)
            #     res=torch.tensor(t[0],device="cuda",dtype=torch.float)
            #     taskid=list(t[1].keys())
            #     pref=torch.tensor([t[1][j] for j in taskid],device="cuda")
            #     new_states[0].append(res)
            #     new_states[1].append(taskid)
            #     new_states[2].append(pref)
            
            values=critic(torch.stack(states[0])[None],
                        torch.stack(states[2])[None],
                        actions[0][None],
                        actions[1][None],
                        actions[2][None],
                        actions[3][None],
                        actions[4][None],
                        actions[5][None])
            
            # loss=0
            # aloss=0
            # loss+=torch.mean((reward-values[0])**2)
            # aloss-=torch.mean((reward-values[0])*torch.stack(action_prob))
            
            # sloss=loss+aloss
            # sloss.backward()
            # actor_optim.step()
            # actor_optim.zero_grad()
            

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
                values[0],
                torch.stack(action_prob),
                # torch.stack(new_states[0]),
                # torch.stack(new_states[2])
            ))
        
        sample_idx=np.arange(batch_size)
        loss=0
        aloss=0
        # print(replay_buf[0][7].shape)
        inputs=[torch.stack([replay_buf[i][j]for i in sample_idx]) for j in range(11)]
        # print(inputs[8].shape,inputs[9].shape,inputs[10].shape)
        inputs[8]=(inputs[8]-torch.mean(inputs[8]))/(torch.std(inputs[8])+1e-9)
        # new_actions=[[] for _ in range(6)]
        loss+=torch.mean((inputs[8]-inputs[9])**2)
        aloss-=torch.mean((inputs[8]-inputs[9]).detach()*inputs[10])
        
        sloss=loss+aloss
        sloss.backward()
        # loss.backward()
        actor_optim.step()
        actor_optim.zero_grad()
        # critic_optim.step()
        # critic_optim.zero_grad()
        eval_reward=eval_rl(actor)[0]
        if eval_reward>best_reward:
            best_reward=eval_reward
            torch.save(actor.state_dict(),f"saved_models/actor_{epoch}.pth")
            torch.save(critic.state_dict(),f"saved_models/critic_{epoch}.pth")
        bar.set_description(f"reward={eval_reward:.4f} best_reward={best_reward:.4f} aloss={aloss:.4f} loss={loss:.4f}")
        
            
if __name__=="__main__":
    train()


