import torch
from torch import nn
from env2 import EDGE_NODE_NUM,EDGE_CLUSTER_NUM,CLOUD_NODE_NUM,CLOUD_CLUSTER_NUM,HISTORY_NUM,TASK_PER_CLUSTER,RES_MAP,FPS_MAP,ECOST,CCOST,ECCOST,FAILC,EDGE_BW,CLOUD_BW,EDGE_RTT,CLOUD_RTT,EDGE_MEM,CLOUD_MEM,EDGE_GPU_MEM,CLOUD_GPU_MEM


# input size=[HISTORY_NUM,(EDGE_NODE_NUM+CLOUD_NODE_NUM*CLOUD_CLUSTER_NUM),4],[TASK_PER_CLUSTER,3],[HISTORY_NUM,TASK_PER_CLUSTER,(5+5+6+5+2+10+1)]
# output size=[TASK_PER_CLUSTER,5]*6
class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu=nn.ReLU()
        # self.lstm=nn.LSTM((EDGE_NODE_NUM+CLOUD_NODE_NUM*CLOUD_CLUSTER_NUM)*4,(EDGE_NODE_NUM+CLOUD_NODE_NUM*CLOUD_CLUSTER_NUM)*16,1,batch_first=True)
        self.lstm2=nn.LSTM(TASK_PER_CLUSTER*(5+5+6+5+2+10+1),128)
        # self.fc1=nn.Linear((EDGE_NODE_NUM+CLOUD_NODE_NUM*CLOUD_CLUSTER_NUM)*16,128)
        self.fc2=nn.Linear(EDGE_CLUSTER_NUM,128)
        self.fc_res=nn.Linear(256*3,5)
        self.fc_fr=nn.Linear(256*3,5)
        self.fc_estep=nn.Linear(256*3,6)
        self.fc_enode=nn.Linear(256*3,5)
        self.fc_ccluster=nn.Linear(256*3,2)
        self.fc_cnode=nn.Linear(256*3,10)
        
    def forward(self,res,cluster_num,pref,history):
        # print(res.shape,pref.shape)
        # res=torch.reshape(res,[-1,HISTORY_NUM,(EDGE_NODE_NUM+CLOUD_NODE_NUM*CLOUD_CLUSTER_NUM)*4])
        # y,_=self.lstm(res)
        # y=y[:,-1,:]
        # print(y.shape)
        # y=self.fc1(y)
        # y=self.relu(y)
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
    
# input size=[EDGE_CLUSTER_NUM,HISTORY_NUM,(EDGE_NODE_NUM+CLOUD_NODE_NUM*CLOUD_CLUSTER_NUM),4],[EDGE_CLUSTER_NUM,TASK_PER_CLUSTER,3],[EDGE_CLUSTER_NUM,TASK_PER_CLUSTER,5]*6
# output size=[EDGE_CLUSTER_NUM]
class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu=nn.ReLU()
        # self.lstm=nn.LSTM(EDGE_CLUSTER_NUM*(EDGE_NODE_NUM+CLOUD_NODE_NUM*CLOUD_CLUSTER_NUM)*4,EDGE_CLUSTER_NUM*(EDGE_NODE_NUM+CLOUD_NODE_NUM*CLOUD_CLUSTER_NUM)*64,1,batch_first=True)
        # self.fc1=nn.Linear(128*3,128)
        self.fc2=nn.Linear(5*5*6,128)
        self.fc3=nn.Linear(128*EDGE_CLUSTER_NUM*(EDGE_NODE_NUM+CLOUD_NODE_NUM*CLOUD_CLUSTER_NUM),EDGE_CLUSTER_NUM*(EDGE_NODE_NUM+CLOUD_NODE_NUM*CLOUD_CLUSTER_NUM))
        self.fc4=nn.Linear(128,3)
    def forward(self,res,pref,y_res,y_fr,y_estep,y_enode,y_ccluster,y_cnode):
        # res=torch.reshape(res,[-1,HISTORY_NUM,EDGE_CLUSTER_NUM*(EDGE_NODE_NUM+CLOUD_NODE_NUM*CLOUD_CLUSTER_NUM)*4])
        # y1,_=self.lstm(res)
        # y1=y1[:,-1,:]
        # y1=torch.reshape(y1,[-1,EDGE_CLUSTER_NUM,EDGE_NODE_NUM+CLOUD_NODE_NUM*CLOUD_CLUSTER_NUM,64])
        # print(y_ccluster.shape,y_cnode.shape)
        nodes=torch.concat([y_enode,torch.reshape(torch.einsum("bijn,bijm->bijnm",y_ccluster,y_cnode),[-1,EDGE_CLUSTER_NUM,TASK_PER_CLUSTER,CLOUD_NODE_NUM*CLOUD_CLUSTER_NUM])],dim=3)
        y2=torch.reshape(torch.einsum("bijn,bijm,bijo->bijnmo",y_res,y_fr,y_estep),[-1,EDGE_CLUSTER_NUM,TASK_PER_CLUSTER,5*5*6])
        y2=self.fc2(y2)
        y2=self.relu(y2)
        # y1.shape==[-1,EDGE_CLUSTER_NUM,EDGE_NODE_NUM+CLOUD_NODE_NUM*CLOUD_CLUSTER_NUM,128]
        # y2.shape==[-1,EDGE_CLUSTER_NUM,TASK_PER_CLUSTER,128]
        # nodes.shape==[-1,EDGE_CLUSTER_NUM,TASK_PER_CLUSTER,EDGE_NODE_NUM+CLOUD_NODE_NUM*CLOUD_CLUSTER_NUM]
        y3=torch.einsum("bijk,bijl->bijlk",y2,nodes)
        # y3.shape==[-1,EDGE_CLUSTER_NUM,TASK_PER_CLUSTER,EDGE_NODE_NUM+CLOUD_NODE_NUM*CLOUD_CLUSTER_NUM,128]
        # y1=torch.broadcast_to(y1[:,:,None,:,:],[-1,-1,TASK_PER_CLUSTER,-1,-1])
        # y1=torch.concat([y1,y3],dim=3)
        # y4=torch.mean(y1,dim=2)
        y4=torch.mean(y3,dim=2)
        y4=self.fc3(torch.reshape(y4,[-1,EDGE_CLUSTER_NUM*(EDGE_NODE_NUM+CLOUD_NODE_NUM*CLOUD_CLUSTER_NUM)*128]))
        y4=torch.reshape(y4,[-1,EDGE_CLUSTER_NUM,(EDGE_NODE_NUM+CLOUD_NODE_NUM*CLOUD_CLUSTER_NUM)])
        y5=torch.einsum("bik,bijkl->bijl",y4,y3)
        y5=self.fc4(y5)
        y5=torch.sum(y5*pref,dim=[2,3])
        return y5
    

    