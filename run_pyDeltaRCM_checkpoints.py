from pyDeltaRCM import BmiDelta
import numpy as np
from scipy.sparse import lil_matrix


load_saved_data = False
save_data = True
saving_dt = 10 # Save every X timesteps


if __name__ == '__main__':
    delta = BmiDelta()
    delta.initialize('deltaRCM.yaml')
    
    
    if load_saved_data:
        delta._delta.uw = np.load('deltaRCM_Output/uw.npy')
        delta._delta.ux = np.load('deltaRCM_Output/ux.npy')
        delta._delta.uy = np.load('deltaRCM_Output/uy.npy')
        delta._delta.depth = np.load('deltaRCM_Output/depth.npy')
        delta._delta.stage = np.load('deltaRCM_Output/stage.npy')
        delta._delta.eta = np.load('deltaRCM_Output/eta.npy')
        
        delta._delta.init_eta = delta._delta.eta.copy()
        delta._delta.strata_eta = lil_matrix(
            np.load('deltaRCM_Output/strata_eta.npy'))
        delta._delta.strata_sand_frac = lil_matrix(
            np.load('deltaRCM_Output/strata_sand_frac.npy'))
        
    

    for time in range(0,50):
        
        print('Time:', delta.get_current_time())
        delta.update()
        
        if (time%saving_dt == 0) & save_data:
            
            print('Overwriting data')
            
            np.save('deltaRCM_Output/uw.npy', delta._delta.uw)
            np.save('deltaRCM_Output/ux.npy', delta._delta.ux)
            np.save('deltaRCM_Output/uy.npy', delta._delta.uy)
            np.save('deltaRCM_Output/depth.npy', delta._delta.depth)
            np.save('deltaRCM_Output/stage.npy', delta._delta.stage)
            np.save('deltaRCM_Output/eta.npy', delta._delta.eta)

            np.save('deltaRCM_Output/strata_eta.npy', 
                     delta._delta.strata_eta.todense())
            np.save('deltaRCM_Output/strata_sand_frac.npy', 
                     delta._delta.strata_sand_frac.todense())
        
        
    delta.finalize()