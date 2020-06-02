from pyDeltaRCM.deltaRCM_driver import pyDeltaRCM
import os

if __name__ == '__main__':

    delta = pyDeltaRCM(input_file = os.path.join(os.getcwd(), 'tests', 'test.yaml'))

    for time in range(0,50):
        delta.update()

    delta.finalize()
