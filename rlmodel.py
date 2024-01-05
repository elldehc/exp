import torch
from torch import nn
from env2 import EDGE_NODE_NUM,EDGE_CLUSTER_NUM,CLOUD_NODE_NUM,CLOUD_CLUSTER_NUM,HISTORY_NUM,TASK_PER_CLUSTER,RES_MAP,FPS_MAP,ECOST,CCOST,ECCOST,FAILC,EDGE_BW,CLOUD_BW,EDGE_RTT,CLOUD_RTT,EDGE_MEM,CLOUD_MEM,EDGE_GPU_MEM,CLOUD_GPU_MEM


# output size=[TASK_PER_CLUSTER,256*3]
class BackBone(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.relu=nn.ReLU()
        # self.lstm=nn.LSTM((EDGE_NODE_NUM+CLOUD_NODE_NUM*CLOUD_CLUSTER_NUM)*4,(EDGE_NODE_NUM+CLOUD_NODE_NUM*CLOUD_CLUSTER_NUM)*16,1,batch_first=True)
        self.lstm2=nn.LSTM((5+5+6+5+2+10+1),128)
        # self.fc1=nn.Linear((EDGE_NODE_NUM+CLOUD_NODE_NUM*CLOUD_CLUSTER_NUM)*16,128)
        self.fc2=nn.Linear(EDGE_CLUSTER_NUM,64)
        self.fc3=nn.Linear(TASK_PER_CLUSTER,64)
        self.fc4=nn.Linear(TASK_PER_CLUSTER,64)
    def forward(self,tasknum,cluster_num,pref,history):
        history=torch.reshape(history,[HISTORY_NUM,TASK_PER_CLUSTER,(5+5+6+5+2+10+1)])
        y1=self.lstm2(history)[0]
        # print(y1.shape)
        y1=torch.mean(y1,dim=[0,1])
        # y1,_=self.lstm2(history)
        # y1=y1[:,-1,:]
        # cluster_num=torch.reshape(cluster_num,[-1,EDGE_CLUSTER_NUM])
        y2=self.fc2(cluster_num)
        y2=self.relu(y2)
        # tasknum=torch.reshape(tasknum,[-1,TASK_PER_CLUSTER])
        y3=self.fc3(tasknum)
        # print(y1.shape,y2.shape,y3.shape)
        y=torch.concat([y1,y2,y3],dim=0)
        # print(pref.shape)
        # pref=torch.reshape(pref,[-1,pref.shape[0],3])
        y=torch.einsum("n,lm->lnm",y,pref)
        y=torch.reshape(y,[y.shape[0],y.shape[1]*y.shape[2]])
        return y
# input size=[HISTORY_NUM,(EDGE_NODE_NUM+CLOUD_NODE_NUM*CLOUD_CLUSTER_NUM),4],[TASK_PER_CLUSTER,3],[HISTORY_NUM,TASK_PER_CLUSTER,(5+5+6+5+2+10+1)]
# output size=[TASK_PER_CLUSTER,5]*6
class ActorHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_res=nn.Linear(256*3,5)
        self.fc_fr=nn.Linear(256*3,5)
        self.fc_estep=nn.Linear(256*3,6)
        self.fc_enode=nn.Linear(256*3,5)
        self.fc_ccluster=nn.Linear(256*3,2)
        self.fc_cnode=nn.Linear(256*3,10)
        
    def forward(self,backbone_y):
        y=backbone_y
        # print(y.shape)
        y_res=self.fc_res(y)
        y_res=nn.Softmax(1)(y_res)
        y_fr=self.fc_fr(y)
        y_fr=nn.Softmax(1)(y_fr)
        y_estep=self.fc_estep(y)
        y_estep=nn.Softmax(1)(y_estep)
        y_enode=self.fc_enode(y)
        y_enode=nn.Softmax(1)(y_enode)
        y_ccluster=self.fc_ccluster(y)
        y_ccluster=nn.Softmax(1)(y_ccluster)
        y_cnode=self.fc_cnode(y)
        y_cnode=nn.Softmax(1)(y_cnode)
        return y_res,y_fr,y_estep,y_enode,y_ccluster,y_cnode
    
# input size=[EDGE_CLUSTER_NUM,TASK_PER_CLUSTER,3],[EDGE_CLUSTER_NUM,TASK_PER_CLUSTER,5]*6
# output size=[EDGE_CLUSTER_NUM]
class CriticHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu=nn.ReLU()
        # self.lstm=nn.LSTM(EDGE_CLUSTER_NUM*(EDGE_NODE_NUM+CLOUD_NODE_NUM*CLOUD_CLUSTER_NUM)*4,EDGE_CLUSTER_NUM*(EDGE_NODE_NUM+CLOUD_NODE_NUM*CLOUD_CLUSTER_NUM)*64,1,batch_first=True)
        self.fc1=nn.Linear(256*3+5*5*6,128)
        self.fc3=nn.Linear(128,1)
        self.fc4=nn.Linear(128,1)
        self.fc5=nn.Linear(128,1)
        self.fc6=nn.Linear(128,1)
        
    def forward(self,clusternum,tasknum,backbone_y,y_res,y_fr,y_estep,y_enode,y_ccluster,y_cnode):
        y32_all=[]
        y51_all=[]
        y61_all=[]
        for i in range(clusternum):
            # print(y_ccluster.shape,y_cnode.shape)
            y_cnodes=torch.einsum("jn,jm->jnm",y_ccluster[i],y_cnode[i])
            y_cnodes=torch.reshape(y_cnodes,[tasknum[i],CLOUD_NODE_NUM*CLOUD_CLUSTER_NUM])
            nodes=torch.concat([y_enode[i],y_cnodes],dim=1)
            y2=torch.reshape(torch.einsum("jn,jm,jo->jnmo",y_res[i],y_fr[i],y_estep[i]),[tasknum[i],5*5*6])
            # y2.shape==[-1,EDGE_CLUSTER_NUM,TASK_PER_CLUSTER,5*5*6]
            # backbone_y.shape==[-1,EDGE_CLUSTER_NUM,TASK_PER_CLUSTER,256*3]
            # nodes.shape==[-1,EDGE_CLUSTER_NUM,TASK_PER_CLUSTER,EDGE_NODE_NUM+CLOUD_NODE_NUM*CLOUD_CLUSTER_NUM]
            y2=torch.concat([y2,backbone_y[i]],dim=1)
            y2=self.fc1(y2)
            y2=self.relu(y2)
            y31=torch.einsum("jk,jl->jlk",y2,y_enode[i])
            y32=torch.einsum("jk,jl->jlk",y2,y_cnodes)
            # y3.shape==[-1,EDGE_CLUSTER_NUM,TASK_PER_CLUSTER,EDGE_NODE_NUM+CLOUD_NODE_NUM*CLOUD_CLUSTER_NUM,128]
            y41=torch.mean(y31,dim=1)
            y32_all.append(y32)
            # y4.shape==[-1,EDGE_CLUSTER_NUM,EDGE_NODE_NUM+CLOUD_NODE_NUM*CLOUD_CLUSTER_NUM,128]
            y51=self.fc3(y41)[:,0]
            y51=nn.Sigmoid()(y51)
            y61=self.fc4(y41)[:,0]
            y51_all.append(y51)
            y61_all.append(y61)
            # print(clusternum,tasknum,y31.shape,y41.shape,y51.shape,y61.shape)
        y32=torch.concat(y32_all)
        y51=torch.concat(y51_all)
        y61=torch.concat(y61_all)
        y42=torch.sum(y32,dim=1)
        y52=self.fc5(y42)[...,0]
        y52=nn.Sigmoid()(y52)
        y62=self.fc6(y42)[...,0]
        # print(clusternum,tasknum,y32.shape,y42.shape,y51.shape,y52.shape,y61.shape,y62.shape)
        y=(1-y51*y52[None])*FAILC+y51*y52[None]*(y61+y62[None])
        return y
    
class ActorCritic(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.backbone=BackBone()
        self.actors=nn.ModuleList()
        for i in range(EDGE_CLUSTER_NUM):
            self.actors.append(ActorHead())
        self.critic=CriticHead()
    def forward(self,clusternum,tasknum,pref,history):
        y_res=[]
        y_fr=[]
        y_estep=[]
        y_enode=[]
        y_ccluster=[]
        y_cnode=[]
        backs=[]
        for i in range(clusternum):
            tasknum_vec=torch.zeros([TASK_PER_CLUSTER],device="cuda",dtype=torch.float)
            tasknum_vec[tasknum[i]-1]=1
            cluster=torch.zeros([EDGE_CLUSTER_NUM],device="cuda",dtype=torch.float)
            cluster[i]=1
            # print(tasknum.shape)
            backs.append(self.backbone(tasknum_vec,cluster,pref[i],history[i]))
            action=self.actors[i](backs[i])
            y_res.append(action[0])
            y_fr.append(action[1])
            y_estep.append(action[2])
            y_enode.append(action[3])
            y_ccluster.append(action[4])
            y_cnode.append(action[5])
        # for i in range(clusternum,EDGE_CLUSTER_NUM):
        #     backs.append(torch.zeros_like(backs[0],device="cuda",dtype=torch.float))
        #     y_res.append(torch.zeros_like(y_res[0],device="cuda",dtype=torch.float))
        #     y_fr.append(torch.zeros_like(y_fr[0],device="cuda",dtype=torch.float))
        #     y_estep.append(torch.zeros_like(y_estep[0],device="cuda",dtype=torch.float))
        #     y_enode.append(torch.zeros_like(y_enode[0],device="cuda",dtype=torch.float))
        #     y_ccluster.append(torch.zeros_like(y_ccluster[0],device="cuda",dtype=torch.float))
        #     y_cnode.append(torch.zeros_like(y_cnode[0],device="cuda",dtype=torch.float))
        # back=torch.stack(backs,dim=1)
        # y_res=torch.stack(y_res,dim=1)
        # y_fr=torch.stack(y_fr,dim=1)
        # y_estep=torch.stack(y_estep,dim=1)
        # y_enode=torch.stack(y_enode,dim=1)
        # y_ccluster=torch.stack(y_ccluster,dim=1)
        # y_cnode=torch.stack(y_cnode,dim=1)
        score=self.critic(clusternum,tasknum,backs,y_res,y_fr,y_estep,y_enode,y_ccluster,y_cnode)
        return y_res,y_fr,y_estep,y_enode,y_ccluster,y_cnode,score