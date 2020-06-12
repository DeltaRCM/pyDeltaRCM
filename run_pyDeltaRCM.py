from pyDeltaRCM.model import DeltaModel
import os

if __name__ == '__main__':

    delta = DeltaModel()

    for time in range(0, 1):
        delta.update()

    delta.finalize()
