import numpy as np
import scipy as sp
from PIL import Image, ImageDraw
import netCDF4

vertical_spacing = 0.05 # in meters
max_depth_of_section = 5 # meters

fp = 'deltaRCM_Output/pyDeltaRCM_output.nc'
nc = netCDF4.Dataset(fp)

strata_sf = nc.variables['strata_sand_frac'][:]
strata_depth = nc.variables['strata_depth'][:]


# shortcuts for array sizes
dz, dx, dy = strata_depth.shape
nx, ny, nz = dx, dy, int(5/vertical_spacing)

# preserves only the oldest surface when cross-cutting
strata = np.zeros_like(strata_depth)
strata[-1,:,:] = strata_depth[-1,:,:]

for i in range(1,dz):
    strata[-i-1,:,:] = np.minimum(strata_depth[-i-1,:,:], strata[-i,:,:])


# combines depths and sand fractions into stratigraphy
stratigraphy = np.zeros((nz, nx, ny))

for j in range(dx):

    mask = np.ones((nz,ny)) * -1

    for i in np.arange(dz-1,-1,-1):

        seds = strata[i,j,:] + max_depth_of_section

        sf = strata_sf[i,j,:]
        sf[sf<0] = 0

        poly = list(zip(np.arange(ny), seds / vertical_spacing)) + list(zip(np.arange(ny)*2, np.arange(ny)*0))

        img = Image.new("L", [ny, nz], 0)
        ImageDraw.Draw(img).polygon(poly, outline=1, fill=1)
        img = np.flipud(img).astype(float)
        
        img *= sf
        mask[img > 0] = img[img > 0]
        
    stratigraphy[:,j,:] = mask
    
print('Saving stratigraphy...')  
np.save('deltaRCM_Output/stratigraphy.npy', stratigraphy)
print('Done')