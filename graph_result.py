import pickle
from matplotlib import pyplot as plt
import numpy as np


methods=["rong","jcab","rl"]
ans=dict()
acc=dict()
lat=dict()
cost=dict()
u=dict()
for it in methods:
    ans[it]=pickle.load(open(f"result_{it}.pkl","rb"))
    acc[it]=sorted([jt[0] for jt in ans[it]])
    lat[it]=sorted([jt[1] for jt in ans[it]])
    cost[it]=sorted([jt[2] for jt in ans[it]])
    u[it]=sorted([jt[3] for jt in ans[it]])


plt.figure()
for it in methods:
    plt.plot(acc[it],np.arange(len(ans[it]),dtype=np.float64)/len(ans[it]),label=it)
plt.ylim((0,1))
plt.xlabel("accuracy")
plt.ylabel("cdf")
plt.legend()
plt.savefig("acc.png")

plt.figure()
for it in methods:
    plt.plot(lat[it],np.arange(len(ans[it]),dtype=np.float64)/len(ans[it]),label=it)
plt.ylim((0,1))
plt.xlabel("latency")
plt.ylabel("cdf")
plt.legend()
plt.savefig("lat.png")

plt.figure()
for it in methods:
    plt.plot(cost[it],np.arange(len(ans[it]),dtype=np.float64)/len(ans[it]),label=it)
plt.ylim((0,1))
plt.xlabel("cost")
plt.ylabel("cdf")
plt.legend()
plt.savefig("cost.png")

plt.figure()
for it in methods:
    plt.plot(u[it],np.arange(len(ans[it]),dtype=np.float64)/len(ans[it]),label=it)
plt.ylim((0,1))
plt.xlim((-1,1))
plt.xlabel("utility")
plt.ylabel("cdf")
plt.legend()
plt.savefig("utility.png")