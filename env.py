import sqlite3
import random
import numpy as np
# from rlmodel import ECOST,CCOST,ECCOST,EDGE_NODE_NUM,CLOUD_NODE_NUM,FAILC,HISTORY_NUM


EDGE_NODE_NUM=5
CLOUD_NODE_NUM=2
HISTORY_NUM=10
RES_MAP = [360, 540, 720, 900, 1080]
FPS_MAP = [2, 3, 5, 10, 15]
ECOST=1.0
CCOST=[2.2,1.8]
ECCOST=0.01
FAILC=-100
EDGE_BW=10*1e9
CLOUD_BW=1e9
EDGE_RTT=0.1
CLOUD_RTT=10
EDGE_MEM=32*1e9
CLOUD_MEM=256*1e9
EDGE_GPU_MEM=16*1e9
CLOUD_GPU_MEM=128*1e9
EDGE_GPU=1
CLOUD_GPU=8

class Env:
    def __init__(self,table_file,tasknum) -> None:
        self.table_file=table_file
        self.tasknum=tasknum
        con=sqlite3.connect("profile_table.db")
        cur=con.cursor()
        res=cur.execute("select * from profile;")
        tablel=res.fetchall()
        table_dict=dict()
        for it in tablel:
            table_dict[tuple(it[:4])]=it[4:]
        self.videos=dict()
        for it in table_dict.keys():
            if it[0] not in self.videos:
                self.videos[it[0]]=it[1]+1
            elif self.videos[it[0]]<it[1]+1:
                self.videos[it[0]]=it[1]+1
        self.video_list=sorted(self.videos.keys())
        self.table_dict=table_dict
        self.edge_res=[[1,32,64,1000] for i in range(5)]
        self.cloud_res=[[1,64,128,1000] for i in range(2)]
        random.seed(114514)
        self.now_video=[self.video_list[random.randint(0,len(self.video_list)-1)]for i in range(tasknum)]
        self.now_seg=[0]*tasknum
        self.now_para=[(random.random(),-random.random(),-random.random()) for i in range(tasknum)]
        self.history=[]
        for i in range(HISTORY_NUM):
            t=[]
            for j in range(EDGE_NODE_NUM):
                t.append([EDGE_GPU,EDGE_GPU_MEM,EDGE_MEM,EDGE_BW,EDGE_RTT])
            for j in range(CLOUD_NODE_NUM):
                t.append([CLOUD_GPU,CLOUD_GPU_MEM,CLOUD_MEM,CLOUD_BW,CLOUD_RTT])
            self.history.append(t)
    def step(self,strategies):
        assert len(strategies)==self.tasknum
        for i in range(self.tasknum):
            for j in [0,1,2,3,6]:
                strategies[i][j]=np.argmax(strategies[i][j].cpu().detach().numpy())
        print(self.now_video[0],self.now_seg[0],strategies[0][0],strategies[0][1])
        item=self.table_dict[(self.now_video[0],self.now_seg[0],strategies[0][0],strategies[0][1])]
        acc=item[0]
        esize=item[1]
        csize=item[strategies[0][2]+1]
        ecomp=sum(item[16:16+strategies[0][2]])
        ccomp=sum(item[16+strategies[0][2]:21])
        latency=esize/(strategies[0][5]*EDGE_BW)+csize/(strategies[0][8]*CLOUD_BW)+ecomp/strategies[0][4]+ccomp/strategies[0][7]
        cost=ecomp*ECOST+ccomp*CCOST[strategies[0][6]]+csize*ECCOST
        value=self.now_para[0][0]*acc+self.now_para[0][1]*latency+self.now_para[0][2]*cost
        etotmem=[0]*EDGE_NODE_NUM
        etotgpumem=[0]*EDGE_NODE_NUM
        ebw=[0]*EDGE_NODE_NUM
        egpu=[0]*EDGE_NODE_NUM
        ctotmem=[0]*CLOUD_NODE_NUM
        ctotgpumem=[0]*CLOUD_NODE_NUM
        cbw=[0]*CLOUD_NODE_NUM
        cgpu=[0]*CLOUD_NODE_NUM
        for i in range(self.tasknum):
            etotmem[strategies[i][3]]+=sum(item[6:6+strategies[i][2]])
            etotgpumem[strategies[i][3]]+=sum(item[11:11+strategies[i][2]])
            ebw[strategies[i][3]]+=strategies[i][5]
            egpu[strategies[i][3]]+=strategies[i][4]
            ctotmem[strategies[i][6]]+=sum(item[6:6+strategies[i][2]])
            ctotgpumem[strategies[i][6]]+=sum(item[11:11+strategies[i][2]])
            cbw[strategies[i][6]]+=strategies[i][8]
            cgpu[strategies[i][6]]+=strategies[i][7]
        fail=0
        for i in range(EDGE_NODE_NUM):
            if etotmem[i]>EDGE_MEM:
                fail=1
                break
            if etotgpumem[i]>EDGE_GPU_MEM:
                fail=1
                break
            if ebw[i]>1:
                fail=1
                break
            if egpu[i]>EDGE_GPU:
                fail=1
                break
        for i in range(CLOUD_NODE_NUM):
            if ctotmem[i]>CLOUD_MEM:
                fail=1
                break
            if ctotgpumem[i]>CLOUD_GPU_MEM:
                fail=1
                break
            if cbw[i]>1:
                fail=1
                break
            if cgpu[i]>CLOUD_GPU:
                fail=1
                break
        for i in range(self.tasknum):
            self.now_seg[i]+=1
            if self.now_seg[i]>=self.videos[self.now_video[i]]:
                self.now_seg[i]=0
                self.now_video[i]=self.video_list[random.randint(0,len(self.video_list)-1)]
                self.now_para=(random.random(),-random.random(),-random.random())
        if not fail:
            t=[]
            for j in range(EDGE_NODE_NUM):
                t.append([EDGE_GPU-egpu[i],EDGE_GPU_MEM-etotgpumem[i],EDGE_MEM-etotmem[i],EDGE_BW-ebw[i],EDGE_RTT])
            for j in range(CLOUD_NODE_NUM):
                t.append([CLOUD_GPU-cgpu[i],CLOUD_GPU_MEM-ctotgpumem[i],CLOUD_MEM-ctotmem[i],CLOUD_BW-cbw[i],CLOUD_RTT])
            self.history.append(t)
        else:
            self.history.append(self.history[-1])
        self.history=self.history[1:]
        if fail:
            return FAILC
        else:
            return value

        

