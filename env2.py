import sqlite3
import random
import numpy as np
from copy import deepcopy
import pickle


EDGE_NODE_NUM=5
EDGE_CLUSTER_NUM=3
CLOUD_NODE_NUM=10
CLOUD_CLUSTER_NUM=2
TASK_PER_CLUSTER=10
HISTORY_NUM=10
RES_MAP = [360, 540, 720, 900, 1080]
FPS_MAP = [2, 3, 5, 10, 15]
ECOST=1.0/3600*10
CCOST=[2.2/3600,1.8/3600]*10
ECCOST=0.01e-6*10
FAILC=-10
EDGE_BW=10e9/8
CLOUD_BW=10e9/8
EDGE_RTT=0.1e-3
CLOUD_RTT=[10e-3,50e-3]
EDGE_MEM=32*1e9
CLOUD_MEM=256*1e9
EDGE_GPU_MEM=16*1e9
CLOUD_GPU_MEM=128*1e9
DELAY_THR=0.2


class Env:
    def __init__(self,table_file="profile_table.db",is_train=True) -> None:
        self.table_file=table_file
        self.tasknum=TASK_PER_CLUSTER*EDGE_CLUSTER_NUM
        self.is_train=is_train
        if not self.is_train:
            self.alpha=np.array(pickle.load(open("our_alpha.pkl","rb"))[:100],dtype=np.float32)
            self.alpha/=np.sum(self.alpha,axis=1)[:,None]
            self.alpha_num=0
        con=sqlite3.connect(self.table_file)
        cur=con.cursor()
        res=cur.execute("select * from profile;")
        tablel=res.fetchall()
        table_dict=dict()
        for it in tablel:
            table_dict[tuple(it[:4])]=it[4:]
        self.videos=dict()
        self.maxacc=dict()
        for it in table_dict.keys():
            if it[0] not in self.videos:
                self.videos[it[0]]=it[1]+1
            elif self.videos[it[0]]<it[1]+1:
                self.videos[it[0]]=it[1]+1
            if (it[0],it[1]) not in self.maxacc:
                self.maxacc[(it[0],it[1])]=table_dict[it][0]
            elif table_dict[it][0]>self.maxacc[(it[0],it[1])]:
                self.maxacc[(it[0],it[1])]=table_dict[it][0]
        self.video_list=sorted(self.videos.keys())
        self.table_dict=table_dict
        
        # self.edge_res=[[[0,1,1,1] for i in range(EDGE_NODE_NUM)] for j in range(EDGE_CLUSTER_NUM)] # resource: tasknum mem gpumem bw
        # self.cloud_res=[[[0,1,1,1] for i in range(CLOUD_NODE_NUM)] for j in range(CLOUD_CLUSTER_NUM)]
        random.seed(114514)
        if self.is_train:
            self.now_video=[self.video_list[random.randint(0,len(self.video_list)-1)]for i in range(self.tasknum)]
            self.now_seg=[0]*self.tasknum
            self.now_para=np.array([(random.random(),random.random(),random.random()) for i in range(self.tasknum)],dtype=np.float32)
            self.now_para/=np.sum(self.now_para,axis=1)[:,None]
        else:
            self.now_video=[self.video_list[i%len(self.video_list)]for i in range(self.tasknum)]
            self.now_seg=[0]*self.tasknum
            self.now_para=[self.alpha[i] for i in range(self.tasknum)]
            self.alpha_num=self.tasknum
            self.now_video_num=self.tasknum%len(self.video_list)
            for i in range(self.tasknum):
                if self.maxacc[(self.now_video[i],self.now_seg[i])]==0:
                    self.getnextseg(i)

        self.now_cluster=[i for j in range(TASK_PER_CLUSTER) for i in range(EDGE_CLUSTER_NUM)]
        self.history=[]
        for i in range(HISTORY_NUM):
            used_edge_res=[[[0,0,0,0] for i in range(EDGE_NODE_NUM)] for j in range(EDGE_CLUSTER_NUM)]
            used_cloud_res=[[[0,0,0,0] for i in range(CLOUD_NODE_NUM)] for j in range(CLOUD_CLUSTER_NUM)]
            self.history.append((used_edge_res,used_cloud_res))
        self.action_history=[]
        for i in range(HISTORY_NUM):
            action=dict()
            for j in range(TASK_PER_CLUSTER*EDGE_CLUSTER_NUM):
                action[j]=[0,0,0,0,0,0,0]
            self.action_history.append(action)
    
    def get_state(self,edge_cluster):
        cluster_tasks=dict()
        for i in range(self.tasknum):
            if self.now_cluster[i]==edge_cluster:
                cluster_tasks[i]=self.now_para[i]
        history=[]
        for it in self.history[-HISTORY_NUM:]:
            # print([len(it[0][edge_cluster])]+[len(it[1][i]) for i in range(CLOUD_CLUSTER_NUM)])
            history.append(it[0][edge_cluster]+sum([it[1][i] for i in range(CLOUD_CLUSTER_NUM)],[]))
        action_history=[]
        for it in self.action_history[-HISTORY_NUM:]:
            # print(it.keys())
            th=[]
            for jt in cluster_tasks:
                for k in range(6):
                    a=[0]*([5,5,6,5,2,10,1][k])
                    # print(it[jt])
                    a[it[jt][k]]=1
                    th+=a
                th.append(it[jt][6])
            action_history.append(th)


        return history,cluster_tasks,action_history

    def submit_action(self,actions): # actions: dict task id->(resolution,frame rate,edge step,edge select,cloud cluster select,cloud select)
        used_edge_res=[[[0,0,0,0] for i in range(EDGE_NODE_NUM)] for j in range(EDGE_CLUSTER_NUM)]
        used_cloud_res=[[[0,0,0,0] for i in range(CLOUD_NODE_NUM)] for j in range(CLOUD_CLUSTER_NUM)]
        acc=dict()
        latency=dict()
        cost=dict()
        size_client_edge=dict()
        size_edge_cloud=dict()
        for it in actions:
            cluster=self.now_cluster[it]
            video=self.now_video[it]
            seg=self.now_seg[it]
            res=self.table_dict[video,seg,actions[it][0],actions[it][1]]
            acc[it]=res[0]/(self.maxacc[(video,seg)]+1e-9)
            sizes=res[1:6]
            edge_res=sum(res[6:6+actions[it][2]]),sum(res[11:11+actions[it][2]])
            cloud_res=sum(res[6+actions[it][2]:11]),sum(res[11+actions[it][2]:16])
            used_edge_res[cluster][actions[it][3]][0]+=1
            used_edge_res[cluster][actions[it][3]][1]+=edge_res[0]
            used_edge_res[cluster][actions[it][3]][2]+=edge_res[1]
            used_cloud_res[actions[it][4]][actions[it][5]][0]+=1
            used_cloud_res[actions[it][4]][actions[it][5]][1]+=cloud_res[0]
            used_cloud_res[actions[it][4]][actions[it][5]][2]+=cloud_res[1]
            size_client_edge[it]=sizes[0]
            size_edge_cloud[it]=sizes[actions[it][2]] if actions[it][2]<5 else 0
            # assert size_client_edge[it]>=0
            # assert self.now_para[it][1]>=0
            used_edge_res[cluster][actions[it][3]][3]+=np.sqrt(sizes[0]*self.now_para[it][1])
            used_cloud_res[actions[it][4]][actions[it][5]][3]+=np.sqrt(size_edge_cloud[it]*self.now_para[it][1])
        tot_ans=[0 for i in range(EDGE_CLUSTER_NUM)]
        all_ans=[]
        for it in actions:
            cluster=self.now_cluster[it]
            res=self.table_dict[video,seg,actions[it][0],actions[it][1]]
            times=res[16:21]
            edge_calc_lat=sum(times[:actions[it][2]])*used_edge_res[cluster][actions[it][3]][0]
            cloud_calc_lat=sum(times[actions[it][2]:])*used_cloud_res[actions[it][4]][actions[it][5]][0]
            edge_trans_lat=used_edge_res[cluster][actions[it][3]][3]/EDGE_BW/np.sqrt(sizes[0]*self.now_para[it][1])
            if size_edge_cloud[it]>0:
                cloud_trans_lat=used_cloud_res[actions[it][4]][actions[it][5]][3]/CLOUD_BW/np.sqrt(size_edge_cloud[it]*self.now_para[it][1])
            else:
                cloud_trans_lat=0
            cost=ECOST*edge_calc_lat+CCOST[actions[it][4]]*cloud_calc_lat+ECCOST*size_edge_cloud[it]
            lat=self.now_para[it][1]*(edge_calc_lat+cloud_calc_lat+EDGE_RTT+(CLOUD_RTT[actions[it][4]] if size_edge_cloud[it]>0 else 0))+edge_trans_lat+cloud_trans_lat
            real_lat=edge_calc_lat+cloud_calc_lat+EDGE_RTT+(CLOUD_RTT[actions[it][4]] if size_edge_cloud[it]>0 else 0)+(edge_trans_lat+cloud_trans_lat)/self.now_para[it][1]
            u=self.now_para[it][0]*acc[it]-lat-self.now_para[it][2]*cost
            # print(self.now_para[it][0]*acc[it],lat,self.now_para[it][2]*cost)
            if used_edge_res[cluster][actions[it][3]][1]>EDGE_MEM or used_edge_res[cluster][actions[it][3]][1]>EDGE_GPU_MEM or used_cloud_res[actions[it][4]][actions[it][5]][1]>CLOUD_MEM or used_cloud_res[actions[it][4]][actions[it][5]][1]>CLOUD_GPU_MEM:
                u=FAILC
            tot_ans[cluster]+=u/TASK_PER_CLUSTER
            actions[it]=list(actions[it])+[u]
            all_ans.append((acc[it],real_lat,cost,u))

        # print("submitted:",actions.keys())
        self.history.append((used_edge_res,used_cloud_res))
        self.action_history.append(actions)
        if len(self.history)>HISTORY_NUM*10:
            self.history=self.history[-HISTORY_NUM*10:]
            self.action_history=self.action_history[-HISTORY_NUM*10:]

        for i in range(self.tasknum):
            self.now_seg[i]+=1
            if self.now_seg[i]>=self.videos[self.now_video[i]]:
                if self.is_train:
                    self.now_video[i]=self.video_list[random.randint(0,len(self.video_list)-1)]
                    self.now_seg[i]=0
                    self.now_para[i]=(random.random(),random.random(),random.random())
                else:
                    self.now_video_num+=1
                    self.alpha_num+=1
                    if self.now_video_num>=len(self.video_list):
                        self.now_video_num=0
                    if self.alpha_num<len(self.alpha):
                        self.now_video[i]=self.video_list[self.now_video_num]
                        self.now_seg[i]=0
                        self.now_para[i]=self.alpha[self.alpha_num]
                    # if self.alpha_num>=len(self.alpha):
                    #     self.alpha_num=0
                        

        return tot_ans,all_ans
    
    def getnextseg(self,cluster):
        while True:
            self.now_seg[cluster]+=1
            if self.now_seg[cluster]>=self.videos[self.now_video[cluster]]:
                if self.is_train:
                    self.now_video[cluster]=self.video_list[random.randint(0,len(self.video_list)-1)]
                    self.now_seg[cluster]=0
                    new_para=np.array([random.random(),random.random(),random.random()],dtype=np.float32)
                    self.now_para[cluster]=new_para/np.sum(new_para)
                else:
                    if self.alpha_num<len(self.alpha):
                        self.now_video[cluster]=self.video_list[self.now_video_num]
                        self.now_seg[cluster]=0
                        self.now_para[cluster]=self.alpha[self.alpha_num]
                    self.now_video_num+=1
                    self.alpha_num+=1
                    if self.now_video_num>=len(self.video_list):
                        self.now_video_num=0
                    
            if self.maxacc[(self.now_video[cluster],self.now_seg[cluster])]>0 or self.finished():
                break
    
    def finished(self):
        return not self.is_train and self.alpha_num>=len(self.alpha)
        


            
