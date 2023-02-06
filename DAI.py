from ast import List
import depthai as dai

class DAI:

    @staticmethod
    def getDevices():
        devices = []

        print('Searching for all available devices...\n')
        # Query all available devices (USB and POE OAK cameras)
        infos: List[dai.DeviceInfo] = dai.DeviceBootloader.getAllAvailableDevices()

        if len(infos) == 0:
            print("Couldn't find any available devices.")
            exit(-1)


        for info in infos:
            # Converts enum eg. 'XLinkDeviceState.X_LINK_UNBOOTED' to 'UNBOOTED'
            state = str(info.state).split('X_LINK_')[1]

            print(f"Found device '{info.name}', MxId: '{info.mxid}', State: '{state}'")
            with dai.Device(dai.Pipeline(), info) as device:
                # print("Available camera sensors: ", device.getCameraSensorNames())
                ccs = device.getConnectedCameras()
                devices.append({ "cameras": len(ccs), "mxid": info.mxid})

            device.close()

        return devices