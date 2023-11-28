
def ParetoOptimal_2d(a):
    i = 0
    b = []
    while i < len(a):
        j = 0
        while j < len(a):
            if i != j:
                vj1 = a[j][0]
                vj2 = a[j][1]
                vi1 = a[i][0]
                vi2 = a[i][1]
                #                 print vj1, vi1, vj2, vi2
                if (vj1 >= vi1 and vj2 <= vi2) and (vj1 > vi1 or vj2 < vi2):
                    i += 1
                    break
                else:
                    j += 1
                if j == len(a):
                    # print(a[i])
                    b.append(a[i])
                    i += 1
                    break
            else:
                j += 1
                if i == len(a) - 1 and j == len(a):
                    # print(a[i])
                    b.append(a[i])
                    i += 1
    return b


def ParetoOptimal_3d(a):
    i = 0
    b = []
    while i < len(a):
        j = 0
        while j < len(a):
            if i != j:
                vj1 = a[j][0]
                vj2 = a[j][1]
                vj3 = a[j][2]
                vi1 = a[i][0]
                vi2 = a[i][1]
                vi3 = a[i][2]
                #                 print vj1, vi1, vj2, vi2
                if (vj1 >= vi1 and vj2 <= vi2 and vj3 <= vi3) and (vj1 > vi1 or vj2 < vi2 or vj3 < vi3):
                    i += 1
                    break
                else:
                    j += 1
                if j == len(a):
                    # print(a[i])
                    b.append(a[i])
                    i += 1
                    break
            else:
                j += 1
                if i == len(a) - 1 and j == len(a):
                    # print(a[i])
                    b.append(a[i])
                    i += 1
    return b

def ParetoFilter(edge_config_dict_all, cloud_config_dict_all):
    ''':
    :arg excel读取的profile全部数据，需要进行pareto filter
    :return pareto filter后的数据，用于形成基于videostream的item，用于后续调度
            edge_config_list, cloud_config_list 由于同一个视频流具有多个object，
            因此需要将多个object的pareto filter结果合并成一个config列表，

            注意: 整个调度过程这个只需要执行一次
    '''
    # print(edge_config_dict_all)
    edge_res = {}
    edge_config_list = []
    for edge_object in edge_config_dict_all:
        # print(edge_object)
        a = [] # 提取dict中的config, ac, edge_cu
        for edge_profile in edge_config_dict_all[edge_object]:
            # print(edge_config_dict_all[edge_object][edge_profile])
            ac = edge_config_dict_all[edge_object][edge_profile]['ac']
            edge_cu = edge_config_dict_all[edge_object][edge_profile]['edge_cu']
            config = edge_profile
            a.append([ac, edge_cu, edge_profile])
            # bw = edge_config_dict_all
        object_optimal = ParetoOptimal_2d(a)
        # print(len(object_optimal))
        for config in object_optimal:
            if config[2] not in edge_config_list:
                edge_config_list.append(config[2])
    for edge_object in edge_config_dict_all:
        object_res = {}
        for config in edge_config_list:
            object_res.update({config: edge_config_dict_all[edge_object][config]})
        # print(object_res)
        edge_res.update({edge_object: object_res})
    # print(edge_res)
    cloud_res = {}
    cloud_config_list = []
    for cloud_object in cloud_config_dict_all:
        # print(edge_object)
        a = [] # 提取dict中的config, ac, edge_cu
        for cloud_profile in cloud_config_dict_all[cloud_object]:
            # print(edge_config_dict_all[edge_object][edge_profile])
            # print(cloud_profile[-1])
            if cloud_profile[-1] == '0':
                ac = cloud_config_dict_all[cloud_object][cloud_profile]['ac']
                edge_cu = cloud_config_dict_all[cloud_object][cloud_profile]['edge_cu']
                cloud_cu = cloud_config_dict_all[cloud_object][cloud_profile]['cloud_cu']
                bw = cloud_config_dict_all[cloud_object][cloud_profile]['bw']
                # print([ac, bw, cloud_cu, cloud_profile])
                a.append([ac, bw, cloud_cu, cloud_profile])
            # bw = edge_config_dict_all
        object_optimal = ParetoOptimal_3d(a)
        object_res = {}
        # print(object_optimal)
        # print(len(object_optimal))
        for item in object_optimal:
            MIN = 11111111
            for i in range(6):
                # print('1111', item[3])
                # tmp = item[3].split(' ')
                # config = tmp[0]+' '+tmp[1]+' '+str(i)
                config = str(int(item[3])+i).zfill(3)
                if config in cloud_config_dict_all[cloud_object]:
                    bw = cloud_config_dict_all[cloud_object][config]['bw']
                    if bw < MIN:
                        MIN = bw
                        if config not in cloud_config_list:
                            # print(config)
                            cloud_config_list.append(config)
        # print(cloud_config_list)
        # object_res.update({config: cloud_config_dict_all[cloud_object][item[3]]})
    for cloud_object in cloud_config_dict_all:
        object_res = {}
        for config in cloud_config_list:
            object_res.update({config: cloud_config_dict_all[cloud_object][config]})
        # print(object_res)
        cloud_res.update({cloud_object: object_res})
    # print(cloud_res)
    # print(len(cloud_res))
    return edge_res, cloud_res, edge_config_list, cloud_config_list


def getConfig(config_dict):
    config_res = []
    for object in config_dict:
        for config in config_dict[object]:
            # print(config)
            if config not in config_res:
                config_res.append(config)
    return config_res


def ConfigurationVideo(edge_config_dict, cloud_config_dict):
    edge_config = getConfig(edge_config_dict)
    cloud_config = getConfig(cloud_config_dict)

    return edge_config, cloud_config