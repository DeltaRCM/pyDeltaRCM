from pyDeltaRCM import BmiDelta
import numpy as np

if __name__ == '__main__':
    delta = BmiDelta()
    delta.initialize('deltaRCM.yaml')

#     for time in range(0,10):
#         delta.update()

    delta.update_until(3.5)
    delta.update_until(12)
