import depthai as dai

with dai.Device() as device:
    print("USB Speed:", device.getUsbSpeed())