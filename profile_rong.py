import sqlite3
import pandas as pd


con=sqlite3.connect("profile_table.db")
cur=con.cursor()
# tablel=res.fetchall()
cprofile={"configuration":[],"bw":[],"edge_it":[],"cloud_it":[],"edge_cu":[],"cloud_cu":[],"ac":[]}
eprofile={"configuration":[],"bw":[],"edge_it":[],"cloud_it":[],"edge_cu":[],"cloud_cu":[],"ac":[]}

for res in range(5):
    for fr in range(5):
        data=cur.execute("select \
                         avg(acc),\
                        avg(size0),\
                        avg(size1),\
                        avg(size2),\
                        avg(size3),\
                        avg(size4),\
                        avg(gpumem0),\
                        avg(gpumem1),\
                        avg(gpumem2),\
                        avg(gpumem3),\
                        avg(gpumem4),\
                        avg(time0),\
                        avg(time1),\
                        avg(time2),\
                        avg(time3),\
                        avg(time4)\
                          from profile where res=? and fr=?;",(res,fr))
        tablel=data.fetchall()[0]
        for step in range(6):
            bw=tablel[1+step]
            edge_lt=sum(tablel[11:11+step])
            cloud_lt=sum(tablel[11+step:16])
            edge_cu=sum(tablel[6:6+step])
            cloud_cu=sum(tablel[6+step:11])
            ac=tablel[0]
            config=f"{res}{fr}{step}"
            cprofile["configuration"].append(config)
            cprofile["bw"].append(bw)
            cprofile["edge_it"].append(edge_lt)
            cprofile["cloud_it"].append(cloud_lt)
            cprofile["edge_cu"].append(edge_cu)
            cprofile["cloud_cu"].append(cloud_cu)
            cprofile["ac"].append(ac)
            if step==5:
                config=f"{res}{fr}"
                eprofile["configuration"].append(config)
                eprofile["bw"].append(bw)
                eprofile["edge_it"].append(edge_lt)
                eprofile["cloud_it"].append(cloud_lt)
                eprofile["edge_cu"].append(edge_cu)
                eprofile["cloud_cu"].append(cloud_cu)
                eprofile["ac"].append(ac)

cdf=pd.DataFrame.from_dict(cprofile)
edf=pd.DataFrame.from_dict(eprofile)
with pd.ExcelWriter("new_cloud_profile.xlsx",mode="a",if_sheet_exists="replace")as f:
    cdf.to_excel(f,"Car",index=False)
    cdf.to_excel(f,"Pes",index=False)
with pd.ExcelWriter("new_edge_profile.xlsx",mode="a",if_sheet_exists="replace")as f:
    edf.to_excel(f,"Car",index=False)
    edf.to_excel(f,"Pes",index=False)
