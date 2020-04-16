from pyDeltaRCM import pyDeltaRCM

if __name__ == '__main__':

    delta = pyDeltaRCM('deltaRCM.yaml')

    for time in range(0,5000):
        delta.update()

    delta.finalize()
