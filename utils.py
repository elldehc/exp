import pandas as pd
import json


RES_MAP = {'360': '0', '600': '1', '720': '2', '900':'3', '1080':'4'}
FPS_MAP = {0: 2, 1: 3, 2: 5, 3: 10, 4: 15}
Interval_time = 10

def excel_to_dict_cloud(file_name):
    '''
    逐个读取excel中每个sheet，将其转化为dict
    :param file_name:
    :return: excel保存为dict形式返回，
                dict格式 {'object name': {config: profile_data, ..., config: profile_data}}
             bw 带宽占用
             edge_it edge推断时间
             cloud_it cloud推断时间
             edge_cu edge计算占用
             cloud_cu cloud计算占用
             ac: 准确度
    '''
    res_dict = {}
    car_dict = {}
    orginal_config = []
    df = pd.read_excel(file_name, sheet_name='Car')
    for i in df.index.values:
        config = df.loc[i][0]
        config = str(int(config)).zfill(3)
        orginal_config.append(config)
        # print(config)
        df_line = df.loc[i, ['bw', 'edge_it', 'cloud_it','edge_cu', 'cloud_cu', 'ac']].to_dict()
        # 将每一行转换成字典后添加到列表
        car_dict.update({config : df_line})
        # print(df_line)
    res_dict.update({'car':car_dict})
    pes_dict = {}
    df = pd.read_excel(file_name, sheet_name='Pes')
    for i in df.index.values:
        config = df.loc[i][0]
        config = str(int(config)).zfill(3)
        df_line = df.loc[i, ['bw', 'edge_it', 'cloud_it', 'edge_cu', 'cloud_cu', 'ac']].to_dict()
        # 将每一行转换成字典后添加到列表
        pes_dict.update({config: df_line})
        # print(df_line)
    # print(pes_dict)
    res_dict.update({'pes': pes_dict})
    return res_dict, orginal_config

def excel_to_dict_edge(file_name):
    '''
    逐个读取excel中每个sheet，将其转化为dict
    :param file_name:
    :return: excel保存为dict形式返回，
                dict格式 {'object name': {config: profile_data, ..., config: profile_data}}
             bw 带宽占用
             edge_it edge推断时间
             cloud_it cloud推断时间
             edge_cu edge计算占用
             cloud_cu cloud计算占用
             ac: 准确度
    '''
    res_dict = {}
    car_dict = {}
    orginal_config = []
    df = pd.read_excel(file_name, sheet_name='Car')
    for i in df.index.values:
        config = df.loc[i][0]
        config = str(int(config)).zfill(2)
        orginal_config.append(config)
        # print(config)
        df_line = df.loc[i, ['bw', 'edge_it', 'cloud_it','edge_cu', 'cloud_cu', 'ac']].to_dict()
        # 将每一行转换成字典后添加到列表
        car_dict.update({config : df_line})
        # print(df_line)
    res_dict.update({'car':car_dict})
    pes_dict = {}
    df = pd.read_excel(file_name, sheet_name='Pes')
    for i in df.index.values:
        config = df.loc[i][0]
        config = str(int(config)).zfill(2)
        df_line = df.loc[i, ['bw', 'edge_it', 'cloud_it', 'edge_cu', 'cloud_cu', 'ac']].to_dict()
        # 将每一行转换成字典后添加到列表
        pes_dict.update({config: df_line})
        # print(df_line)
    # print(pes_dict)
    res_dict.update({'pes': pes_dict})
    return res_dict, orginal_config

def read_json(filename):
    file = open(filename, 'r')
    string_camera_info = file.read()
    camera_info = json.loads(string_camera_info)
    file.close()
    # print(camera_info)
    return camera_info

'''
solution输出格式：
{camera 1 : {'resource':[edge_cu, cloud_cu, bw], 'utility': U, 'loc' : [0,1] or [1, 0] or [1, 1],  
appid: [config, location, migration_flag],..., {appid: [config, location, migration_flag]}, 
'resource': resource, 'utility': utility, 'loc': loc, 'config': config}

camera 应用：
{cameraid: {appid: [utilty_function, objectid],...},..., cameraid} : 每个cameraid对应一个dict, 
dict保存app信息，每个appid包含使用的utility function和关注的待检测的objectid

objectid 0: car 1: pes

'''
'''
新应用的抵达，burst持续10-300s之间，拥有较高的quality和较低的延迟要求
网络带宽发生了变化 
'''


