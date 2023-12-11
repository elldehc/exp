import torch
from torch import nn
from env2 import EDGE_NODE_NUM,EDGE_CLUSTER_NUM,CLOUD_NODE_NUM,CLOUD_CLUSTER_NUM,HISTORY_NUM,TASK_PER_CLUSTER,RES_MAP,FPS_MAP,ECOST,CCOST,ECCOST,FAILC,EDGE_BW,CLOUD_BW,EDGE_RTT,CLOUD_RTT,EDGE_MEM,CLOUD_MEM,EDGE_GPU_MEM,CLOUD_GPU_MEM


# output size=[TASK_PER_CLUSTER,256*3]
class BackBone(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.relu=nn.ReLU()
        # self.lstm=nn.LSTM((EDGE_NODE_NUM+CLOUD_NODE_NUM*CLOUD_CLUSTER_NUM)*4,(EDGE_NODE_NUM+CLOUD_NODE_NUM*CLOUD_CLUSTER_NUM)*16,1,batch_first=True)
        self.lstm2=nn.LSTM(TASK_PER_CLUSTER*(5+5+6+5+2+10+1),128)
        # self.fc1=nn.Linear((EDGE_NODE_NUM+CLOUD_NODE_NUM*CLOUD_CLUSTER_NUM)*16,128)
        self.fc2=nn.Linear(EDGE_CLUSTER_NUM,128)
    def forward(self,cluster_num,pref,history):
        history=torch.reshape(history,[-1,HISTORY_NUM,TASK_PER_CLUSTER*(5+5+6+5+2+10+1)])
        y1,_=self.lstm2(history)
        y1=y1[:,-1,:]
        cluster_num=torch.reshape(cluster_num,[-1,EDGE_CLUSTER_NUM])
        y2=self.fc2(cluster_num)
        y2=self.relu(y2)
        y=torch.concat([y1,y2],dim=1)
        # print(pref.shape)
        pref=torch.reshape(pref,[-1,TASK_PER_CLUSTER,3])
        y=torch.einsum("bn,blm->blnm",y,pref)
        y=torch.reshape(y,[-1,y.shape[1],y.shape[2]*y.shape[3]])
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
    def forward(self,backbone_y,y_res,y_fr,y_estep,y_enode,y_ccluster,y_cnode):
        nodes=torch.concat([y_enode,torch.reshape(torch.einsum("bijn,bijm->bijnm",y_ccluster,y_cnode),[-1,EDGE_CLUSTER_NUM,TASK_PER_CLUSTER,CLOUD_NODE_NUM*CLOUD_CLUSTER_NUM])],dim=3)
        y2=torch.reshape(torch.einsum("bijn,bijm,bijo->bijnmo",y_res,y_fr,y_estep),[-1,EDGE_CLUSTER_NUM,TASK_PER_CLUSTER,5*5*6])
        # y2.shape==[-1,EDGE_CLUSTER_NUM,TASK_PER_CLUSTER,5*5*6]
        # backbone_y.shape==[-1,EDGE_CLUSTER_NUM,TASK_PER_CLUSTER,256*3]
        # nodes.shape==[-1,EDGE_CLUSTER_NUM,TASK_PER_CLUSTER,EDGE_NODE_NUM+CLOUD_NODE_NUM*CLOUD_CLUSTER_NUM]
        y2=torch.concat([y2,backbone_y],dim=3)
        y2=self.fc1(y2)
        y2=self.relu(y2)
        y3=torch.einsum("bijk,bijl->bijlk",y2,nodes)
        # y3.shape==[-1,EDGE_CLUSTER_NUM,TASK_PER_CLUSTER,EDGE_NODE_NUM+CLOUD_NODE_NUM*CLOUD_CLUSTER_NUM,128]
        y4=torch.mean(y3,dim=2)
        # y4.shape==[-1,EDGE_CLUSTER_NUM,EDGE_NODE_NUM+CLOUD_NODE_NUM*CLOUD_CLUSTER_NUM,128]
        y5=self.fc3(y4)
        y5=nn.Sigmoid()(y5)
        y6=self.fc4(y4)
        y5=y5*FAILC+(1-y5)*y6
        y5=torch.mean(y5,dim=[2,3])
        return y5
    
class ActorCritic(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.backbone=BackBone()
        self.actors=nn.ModuleList()
        for i in range(EDGE_CLUSTER_NUM):
            self.actors.append(ActorHead())
        self.critic=CriticHead()
    def forward(self,pref,history):
        y_res=[]
        y_fr=[]
        y_estep=[]
        y_enode=[]
        y_ccluster=[]
        y_cnode=[]
        backs=[]
        for i in range(EDGE_CLUSTER_NUM):
            cluster=torch.zeros([EDGE_CLUSTER_NUM],device="cuda",dtype=torch.float)
            cluster[i]=1
            backs.append(self.backbone(cluster,pref[i],history[i]))
            action=self.actors[i](backs[i])
            y_res.append(action[0])
            y_fr.append(action[1])
            y_estep.append(action[2])
            y_enode.append(action[3])
            y_ccluster.append(action[4])
            y_cnode.append(action[5])
        back=torch.stack(backs,dim=1)
        y_res=torch.stack(y_res,dim=1)
        y_fr=torch.stack(y_fr,dim=1)
        y_estep=torch.stack(y_estep,dim=1)
        y_enode=torch.stack(y_enode,dim=1)
        y_ccluster=torch.stack(y_ccluster,dim=1)
        y_cnode=torch.stack(y_cnode,dim=1)
        score=self.critic(back,y_res,y_fr,y_estep,y_enode,y_ccluster,y_cnode)
        return y_res,y_fr,y_estep,y_enode,y_ccluster,y_cnode,score