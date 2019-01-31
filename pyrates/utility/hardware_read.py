from tensorflow.python.client import device_lib
# import tensorflow-gpu

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


def get_available_cpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'CPU']


if __name__ == "__main__":
    # devices = device_lib.list_local_devices()
    # for device in devices:
    #     print(device)
    # cpu = get_available_cpus()
    # print(type(*cpu))

    print(get_available_gpus())
    print(get_available_cpus())
