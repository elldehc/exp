import pickle
from matplotlib import pyplot as plt
import numpy as np


ans_rong=pickle.load(open("result_rong.pkl","rb"))
ans_rl=pickle.load(open("result_rl.pkl","rb"))

acc_rong=sorted([it[0] for it in ans_rong])
lat_rong=sorted([it[1] for it in ans_rong])
cost_rong=sorted([it[2] for it in ans_rong])
u_rong=sorted([it[3] for it in ans_rong])
acc_rl=sorted([it[0] for it in ans_rl])
lat_rl=sorted([it[1] for it in ans_rl])
cost_rl=sorted([it[2] for it in ans_rl])
u_rl=sorted([it[3] for it in ans_rl])

plt.figure()
plt.plot(np.arange(len(ans_rong),dtype=np.float64)/len(ans_rong),acc_rong,label="rong")
plt.plot(np.arange(len(ans_rl),dtype=np.float64)/len(ans_rl),acc_rl,label="ours")
plt.ylim((0,1))
plt.xlabel("accuracy")
plt.ylabel("cdf")
plt.legend()
plt.savefig("acc.png")

plt.figure()
plt.plot(np.arange(len(ans_rong),dtype=np.float64)/len(ans_rong),lat_rong,label="rong")
plt.plot(np.arange(len(ans_rl),dtype=np.float64)/len(ans_rl),lat_rl,label="ours")
plt.ylim((0,1))
plt.xlabel("latency")
plt.ylabel("cdf")
plt.legend()
plt.savefig("lat.png")

plt.figure()
plt.plot(np.arange(len(ans_rong),dtype=np.float64)/len(ans_rong),cost_rong,label="rong")
plt.plot(np.arange(len(ans_rl),dtype=np.float64)/len(ans_rl),cost_rl,label="ours")
plt.ylim((0,1))
plt.xlabel("cost")
plt.ylabel("cdf")
plt.legend()
plt.savefig("cost.png")

plt.figure()
plt.plot(np.arange(len(ans_rong),dtype=np.float64)/len(ans_rong),u_rong,label="rong")
plt.plot(np.arange(len(ans_rl),dtype=np.float64)/len(ans_rl),u_rl,label="ours")
plt.ylim((0,1))
plt.xlabel("utility")
plt.ylabel("cdf")
plt.legend()
plt.savefig("utility.png")