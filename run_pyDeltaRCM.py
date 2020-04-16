from pyDeltaRCM.deltaRCM_driver import pyDeltaRCM

if __name__ == '__main__':

    delta = pyDeltaRCM('deltaRCM.yaml')

    for time in range(0,1):
        delta.update()

    delta.finalize()
