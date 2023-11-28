import os
import random
import json
def assign_app_ID(index):
    return '%04d'%(index)
def assign_camera_ID(index):
    return '%04d'%(index)

def CreateCameraApp(camera_num, app_num, func_utility_num, object_num, prefer_num, action, camera, index):

    if action=='r':
        for i in range(camera_num):
            camera.update({assign_camera_ID(i): {}})
        tmp_prefer  = [0 for i in range(app_num)]

        cur_prefer = 1
        while cur_prefer < prefer_num:
            tmp_cur_prefer = []
            cnt = 0
            while cnt < app_num/prefer_num:
                tmp_index = random.randint(0, app_num-1)
                # print(tmp_index)
                if tmp_index in tmp_cur_prefer:
                    continue
                else:
                    if tmp_prefer[tmp_index] == 0:
                        tmp_cur_prefer.append(tmp_index)
                        tmp_prefer[tmp_index] = cur_prefer
                        cnt = cnt+1
            cur_prefer = cur_prefer+1
            # print(len(tmp_cur_prefer))
            # break
        # print(tmp_prefer)
        # print(tmp_prefer.count(0), tmp_prefer.count(1), tmp_prefer.count(2))
        while index < app_num:
            for i in range(camera_num):
                if random.randint(0,1) == 1:
                    cameraid = assign_camera_ID(i)
                    appid = assign_app_ID(index)
                    # print(appid)
                    index = index + 1
                    if index == app_num:
                        break
                    camera[cameraid].update({appid: [random.randint(0, func_utility_num-1), tmp_prefer[index],random.randint(0, object_num-1)]})
                    # 由于预设的utility function包括3个等级，accuracy-prefer, latency-prefer, balance-prefer
    elif action=='m':

        mean_app_num = app_num / camera_num

        tmp_prefer  = [0 for i in range(app_num)]

        cur_prefer = 1
        while cur_prefer < prefer_num:
            tmp_cur_prefer = []
            cnt = 0
            while cnt < app_num/prefer_num:
                tmp_index = random.randint(0, app_num-1)
                # print(tmp_index)
                if tmp_index in tmp_cur_prefer:
                    continue
                else:
                    if tmp_prefer[tmp_index] == 0:
                        tmp_cur_prefer.append(tmp_index)
                        tmp_prefer[tmp_index] = cur_prefer
                        cnt = cnt+1
            cur_prefer = cur_prefer+1
        print(tmp_prefer)
        for i in range(app_num):
            camera_id = i / mean_app_num
            cameraid = assign_camera_ID(camera_id)
            appid = assign_app_ID(i+index)
            camera[cameraid].update({appid: [random.randint(0, func_utility_num-1), tmp_prefer[i],random.randint(0, object_num-1)]})
    index = index+app_num
    return camera, index


def ConfigurationVideo():
    res = [1080, 900, 720, 600, 360]
    fps = [15, 10, 5, 3, 2]
    point = [0, 1, 2, 3, 4, 5, 6]
    edge_config = []
    cloud_config = []
    for r in res:
        for f in fps:
            edge_config.append(str(r)+' '+str(f))
            for p in point:
                cloud_config.append(str(r)+' '+str(f)+' '+str(p))
    # print(edge_config)
    # print(cloud_config)
    return edge_config, cloud_config
        # camera
def CameraEdgeMaping(camera, region_num, camera_num):
    camera_map = [[] for i in range(region_num)]
    camera_maping = {}
    for cid in camera:
        # print(cid)
        # num = cid.split('a')[-1]
        # print(num)
        camera_map[int(int(cid)/(camera_num/region_num))].append(cid)
    for c in camera_map:
        index = camera_map.index(c)
        camera_maping.update({index: c})
    return camera_maping





if __name__ == '__main__':

    camera_num = 50
    app_num = 300  #  需要更改这里
    func_utility_num = 1
    object_num = 2
    prefer_num = 3
    action = 'm'
    camera = {}
    index = 0

    for i in range(camera_num):
        camera.update({assign_camera_ID(i): {}})
    camera, index = CreateCameraApp(camera_num, app_num, func_utility_num, object_num, prefer_num, action, camera, index)
    # print(index)
    file = open('camera_app-{m,50,6}.json', 'w')  # 平均 50个camera, 每个camera 5个app
    file.write(json.dumps(camera, indent=4))
    file.close()

    # camera, index = CreateCameraApp(camera_num, app_num, func_utility_num, object_num, prefer_num, action, camera,
    #                                 index)
    # # print(index)
    # file = open('camera_app-{m,50,2}.json', 'w')  # 平均 50个camera, 每个camera 5个app
    # file.write(json.dumps(camera, indent=4))
    # file.close()
    # camera, index = CreateCameraApp(camera_num, app_num, func_utility_num, object_num, prefer_num, action, camera,
    #                                 index)
    # # print(index)
    # file = open('camera_app-{m,50,3}.json', 'w')  # 平均 50个camera, 每个camera 5个app
    # file.write(json.dumps(camera, indent=4))
    # file.close()
    # camera, index = CreateCameraApp(camera_num, app_num, func_utility_num, object_num, prefer_num, action, camera,
    #                                 index)
    # # print(index)
    # file = open('camera_app-{m,50,4}.json', 'w')  # 平均 50个camera, 每个camera 5个app
    # file.write(json.dumps(camera, indent=4))
    # file.close()
    '''
    path = os.getcwd()
    
    file_name = os.path.join'camera_app-{m,50,4}.json'
    file = open(file_name, 'r')
    string_camera_info = file.read()
    camera = json.loads(string_camera_info)
    index = 200
    camera, index = CreateCameraApp(camera_num, app_num, func_utility_num, object_num, prefer_num, action, camera,
                                    index)

    file = open('camera_app-{m,50,5}.json', 'w')  # 平均 50个camera, 每个camera 5个app
    file.write(json.dumps(camera, indent=4))
    file.close()


    ConfigurationVideo()
    # file = open('init_camera_app-{m,50,2}.json', 'w') # 平均 50个camera, 每个camera 2个app
    # file = open('init_camera_app-{m,50,3}.json', 'w') # 平均 50个camera, 每个camera 3个app
    # file = open('init_camera_app-{m,50,4}.json', 'w') # 平均 50个camera, 每个camera 4个app
    '''
