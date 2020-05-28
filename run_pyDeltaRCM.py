from pyDeltaRCM.model import DeltaModel
import os

if __name__ == '__main__':

    delta = DeltaModel(input_file = os.path.join(os.getcwd(), 'tests', 'test.yaml'))

    for time in range(0, 0):
        delta.update()

    delta.finalize()
