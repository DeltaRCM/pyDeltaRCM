from pyDeltaRCM import BmiDelta
import numpy as np

if __name__ == '__main__':
    delta = BmiDelta()
    delta.initialize('deltaRCM.yaml')
    

    for time in range(0,5000):
        print 'Time:', delta.get_current_time()
        delta.update()
        
    delta.finalize()
        
        
