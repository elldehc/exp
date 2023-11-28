import pynvml


# computation_ability_table = {
#     "NVIDIA GeForce RTX 3060 Laptop GPU": 1000
# }

def query_computation_resource():
    pynvml.nvmlInit()
    gpu_number = pynvml.nvmlDeviceGetCount()
    print("GPU number: " + str(gpu_number))  # how many GPU
    total_resource = 0
    for gpu_id in range(gpu_number):
        total_resource += query_by_id(gpu_id)
    return total_resource


def query_by_id(gpu_id):
    '''
       查询gpuid下显卡使用显存情况
       :param gpu_id: 有效的gpuid，数字类型，无效gpuid则报错.
       :return: 使用显存大小，单位M
    '''
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    gpu_name = pynvml.nvmlDeviceGetName(handle).decode('UTF-8')
    print(gpu_name)

    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
    powerusage = pynvml.nvmlDeviceGetPowerUsage(handle)

    print(utilization)
    return 1000 * (1 - utilization.gpu / 100)


if __name__=='__main__':
    query_computation_resource()
