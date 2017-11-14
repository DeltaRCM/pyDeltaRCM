from pyDeltaRCM import BmiDelta
import numpy as np

if __name__ == '__main__':
    delta = BmiDelta()
    delta.initialize('deltaRCM.yaml')
    

    for time in range(0,1000):
        print 'Timestep:', delta.get_time_step()
        delta.update()
        
        
