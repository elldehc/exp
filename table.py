import sqlite3
import torch
import torch.cuda
from torch import nn
from pathlib import Path
import ffmpeg
import cv2
import numpy as np
import torchvision
from backbone.resnet101 import ResNet101
from model import Model
from dataset.coco2017 import COCO2017
from config.eval_config import EvalConfig as Config
import backbone.base
from typing import Tuple
import pandas as pd
from torchvision.transforms import transforms
from tqdm import tqdm


class ResNet101Back(backbone.base.Base):

    def __init__(self,pretrained: bool):
        super().__init__(pretrained)
        self.split_pos=7

    def features(self) -> Tuple[nn.Module, nn.Module, int, int]:
        resnet101 = torchvision.models.resnet101(pretrained=self._pretrained)

        # list(resnet101.children()) consists of following modules
        #   [0] = Conv2d, [1] = BatchNorm2d, [2] = ReLU,
        #   [3] = MaxPool2d, [4] = Sequential(Bottleneck...),
        #   [5] = Sequential(Bottleneck...),
        #   [6] = Sequential(Bottleneck...),
        #   [7] = Sequential(Bottleneck...),
        #   [8] = AvgPool2d, [9] = Linear
        children = list(resnet101.children())
        bn_features=children[:-3]
        num_features_out = 1024

        hidden = children[-3]
        num_hidden_out = 2048

        # for parameters in [feature.parameters() for i, feature in enumerate(features) if i <= 4]:
        #     for parameter in parameters:
        #         parameter.requires_grad = False

        features = nn.Sequential()

        return features, hidden, num_features_out, num_hidden_out
    
RES_MAP = [360, 540, 720, 900, 1080]
FPS_MAP = [2, 3, 5, 10, 15]
RESNET101_CHECKPOINT="model-180000.pth"
DATA_PATH=Path("/mnt/data20/datasets/VisDrone2019-VID-test-dev")

# resnet101 = torchvision.models.resnet101(pretrained=False)
# resnet101.load_state_dict(torch.load(RESNET101_CHECKPOINT),strict=False)
# children = list(resnet101.children())
# for parameters in [feature.parameters() for i, feature in enumerate(children)]:
#     for parameter in parameters:
#         parameter.requires_grad = False

# model_steps=[nn.Sequential(*children[:4]).cuda(),children[4].cuda(),children[5].cuda(),children[6].cuda()]
model=Model(ResNet101(pretrained=False),COCO2017.num_classes(), pooler_mode=Config.POOLER_MODE,
    anchor_ratios=Config.ANCHOR_RATIOS, anchor_sizes=[64,128,256,512],
    rpn_pre_nms_top_n=Config.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=Config.RPN_POST_NMS_TOP_N,need_profile=True).cuda()
model.load(RESNET101_CHECKPOINT)

# print(list(list(model.detection.children())[0].parameters())[0][:10])
# assert 0
for it in model.parameters():
    it.requires_grad=False
# model.load(RESNET101_CHECKPOINT,strict=False)
model.eval()
# model_steps.append(model)
# for step in model_steps:
#     step.eval()

def calc_acc(detbb,gtbb,threshold=0.5):
    frames=np.max(gtbb[:,0])
    scores=[]
    fp=0
    tp=0
    for i in range(1,frames+1):
        detframe=detbb[detbb[:,0]==i]
        gtframe=gtbb[gtbb[:,0]==i]
        aiou=np.zeros([gtframe.shape[0],detframe.shape[0]])
        ix=np.maximum(0,np.minimum(gtframe[:,None,1]+gtframe[:,None,3],detframe[None,:,1]+detframe[None,:,3])-np.maximum(gtframe[:,None,1],detframe[None,:,1]))
        iy=np.maximum(0,np.minimum(gtframe[:,None,2]+gtframe[:,None,4],detframe[None,:,2]+detframe[None,:,4])-np.maximum(gtframe[:,None,2],detframe[None,:,2]))
        ai=ix*iy
        au=((detframe[:,3])*(detframe[:,4]))[None,:]+((gtframe[:,3])*(gtframe[:,4]))[:,None]-ai
        aiou=ai/au
        if detframe.shape[0]==0:
            continue
        # for j in range(detframe.shape[0]):
        #     for k in range(gtframe.shape[0]):
        #         ix=max(0,min(detframe[j,3],gtframe[k,3])-max(detframe[j,1],gtframe[k,1]))
        #         iy=max(0,min(detframe[j,4],gtframe[k,4])-max(detframe[j,2],gtframe[k,2]))
        #         ai=ix*iy
        #         au=(detframe[j,3]-detframe[j,1])*(detframe[j,4]-detframe[j,2])+(gtframe[j,3]-gtframe[j,1])*(gtframe[j,4]-gtframe[j,2])-ai
        #         if au>0:
        #             aiou[j,k]=ai/au
        # print(aiou.shape)
        prob_sorted=np.argsort(-detframe[:,5])
        gt_sorted=np.argsort(gtframe[:,5])
        matchd=np.zeros([detframe.shape[0]])
        matchg=np.zeros([gtframe.shape[0]])
        for j in range(detframe.shape[0]):
            tj=prob_sorted[j]
            maxoa=threshold
            for k in range(gtframe.shape[0]):
                tk=gt_sorted[k]
                # if tj==0:
                #     print(tk,matchg[tk],aiou[tk,tj],gtframe[tk,5])
                if matchg[tk]:
                    continue
                if gtframe[tk,5]==1 and matchd[tj]!=0:
                    break
                if aiou[tk,tj]<maxoa:
                    continue
                maxoa=aiou[tk,tj]
                if gtframe[tk,5]==1:
                    matchd[tj]=-1
                    matchg[tk]=-1
                else:
                    matchd[tj]=1
                    matchg[tk]=1
        tp+=np.sum(matchd!=0)
        fp+=np.sum(matchd==0)
        # print(ai[:,0])
        # print(aiou[:,0])
        # print(matchd)
        # print(matchg)
        # scores.append(np.sum(matchd==1)/np.sum(matchd!=-1))
        # aiou_sorted=np.argsort(aiou,axis=1)
        # t_scores=np.zeros([detframe.shape[0]])
        # for j in range(gtframe.shape[0]):
        #     t_scores[aiou_sorted[j,0]]=aiou[j,aiou_sorted[j,0]]
        # scores.append(t_scores)    
        
    # scores=np.concatenate(scores)
    # return metrics.average_precision_score(np.ones_like(scores),scores)
    if tp+fp==0:
        return 0
    else:
        return tp/(tp+fp)

# acc, video size, size after step 1-4, mem usage of step 1-5, gpu mem usage of step 1-5, computing time of step 1-5
def calc_row(video_file,seg,res,fr):
    images=[]
    for it in range(1,31):
        im=cv2.imread(str(DATA_PATH/"sequences"/video_file/"{:07d}.jpg".format(seg*30+it)))
        images.append(im)
    images=np.stack(images)
    ori_size=(images.shape[2],images.shape[1])
    # print(ori_size)
    ffmpeg.input("pipe:",
        format="rawvideo",
        pix_fmt="bgr24",
        video_size=(images.shape[2],images.shape[1]),
        framerate=FPS_MAP[fr]
        )\
        .filter("scale",size=f"{RES_MAP[res]//9*16}:{RES_MAP[res]}")\
        .output(f"tmp/{video_file}_{seg}_{res}_{fr}.mp4",vcodec="h264",pix_fmt="rgb24",format="mp4")\
        .overwrite_output()\
        .run(input=np.ravel(images[::30//FPS_MAP[fr]]).tobytes(),quiet=True)
    frames_edge=open(f"tmp/{video_file}_{seg}_{res}_{fr}.mp4","rb").read()
    video_size=len(frames_edge)
    images=torch.tensor(np.transpose(images[::30//FPS_MAP[fr],:,:,[2,1,0]],[0,3,1,2]).astype(np.float32)/256).detach().cuda()
    transform = transforms.Compose([
            # transforms.ToTensor(),
            transforms.Resize((RES_MAP[res],RES_MAP[res]//9*16)),  # interpolation `BILINEAR` is applied by default
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    images = transform(images)
    # images=images.detach().cuda()
    # print(images[0,:,:10,:10])
    sizes=[video_size]
    
    detbb,detc,detprob,detfr,comp_time,inter_results,mem_usage,gpu_mem_usage=model(images)
    for step in range(4):
        frame=inter_results[step].cpu().numpy()
        frame_flattened=np.reshape(frame,[frame.shape[0]*frame.shape[1],frame.shape[2]*frame.shape[3]])
        mx=np.max(np.abs(frame_flattened))
        f=np.math.ceil(np.math.log2(mx/128))
        frame_flattened=(frame_flattened/(2**f)).astype(np.uint8)
        retval,data=cv2.imencode(".png",frame_flattened)
        sizes.append(len(data))
    # detbb,detc,detprob,detfr=images
    detbb=detbb.cpu().numpy()/RES_MAP[res]*ori_size[1]
    detc=detc.cpu().numpy()
    detprob=detprob.cpu().numpy()
    detfr=detfr.cpu().numpy()
    # detbb=np.repeat(detbb,30//FPS_MAP[fr],axis=0)
    # print(detbb.shape,detc.shape,detprob.shape,detfr.shape)
    # print(detbb[:10])
    # print(detc[:10])
    # print(detprob[:10])
    # print(detfr[:10])
    t=detprob>0.6
    detbb=detbb[t]
    detc=detc[t]
    detprob=detprob[t]
    detfr=detfr[t]
    # print(detbb.shape,detc.shape,detprob.shape,detfr.shape)
    # print(detbb[:10])
    # print(detc[:10])
    # print(detprob[:10])
    # print(detfr[:10])
    detbb[:,2]-=detbb[:,0]
    detbb[:,3]-=detbb[:,1]
    newdetbb=np.concatenate([detfr[:,None],detbb,detprob[:,None]],axis=1)
    # print(detbb[detfr==0])
    # newdetbb=np.zeros([detbb.shape[0]*30//FPS_MAP[fr],6])
    # for i in range(detbb.shape[0]):
    #     for j in range(30//FPS_MAP[fr]):
    #         newdetbb[i*30//FPS_MAP[fr]+j,1:5]=detbb[i]
    #         newdetbb[i*30//FPS_MAP[fr]+j,0]=detfr[i]
    #         newdetbb[i*30//FPS_MAP[fr]+j,5]=detprob[i]

    
    df=pd.read_csv(DATA_PATH/"annotations"/(video_file+".txt"),header=None).to_numpy()
    gtbb=df[:,[0,2,3,4,5,6]]
    gtbb[:,0]-=1
    # print(gtbb.shape)
    # print(gtbb[gtbb[:,0]==0,1:5])
    acc=calc_acc(newdetbb,gtbb)

    
    return acc,sizes,mem_usage,gpu_mem_usage,comp_time

if __name__=="__main__":
    con=sqlite3.connect("profile_table-test.db")
    cur=con.cursor()
    cur.execute("create table if not exists profile \
                (video text,\
                seg integer,\
                res integer,\
                fr integer,\
                acc real,\
                size0 integer,\
                size1 integer,\
                size2 integer,\
                size3 integer,\
                size4 integer,\
                mem0 integer,\
                mem1 integer,\
                mem2 integer,\
                mem3 integer,\
                mem4 integer,\
                gpumem0 integer,\
                gpumem1 integer,\
                gpumem2 integer,\
                gpumem3 integer,\
                gpumem4 integer,\
                time0 real,\
                time1 real,\
                time2 real,\
                time3 real,\
                time4 real,\
                primary key (video,seg,res,fr)\
                );")
    videos=[it.name for it in (DATA_PATH/"sequences").iterdir()]
    video_segs=[]
    for video in videos:
        segnum=len(list((DATA_PATH/"sequences"/video).glob("*.jpg")))//30
        for i in range(segnum):
            for j in range(5):
                for k in range(5):
                    video_segs.append((video,i,j,k))
    res=cur.execute("select video,seg,res,fr from profile;")
    done_list=set(res.fetchall())
    for seg in tqdm(video_segs):
        if seg not in done_list:
            ans=calc_row(seg[0],seg[1],seg[2],seg[3])
            flatten_ans=list(seg)+[ans[0]]+ans[1]+ans[2]+ans[3]+ans[4]
            cur.execute("insert into profile values ("+",".join("?"*25)+");",flatten_ans)
            con.commit()
    con.close()


        
