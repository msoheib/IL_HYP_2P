#%%

from scipy import io
import numpy as np

st = np.load('stat.npy', allow_pickle=True)
n_rois = st.shape[0]
# Image size in pixels
# What if motion correction changed the image size? 
height = 512
width = 512
spatial_footprints = np.zeros((n_rois, height, width))

for i in range(n_rois):
    spatial_footprints[i, st[i]['ypix'], st[i]['xpix']] = st[i]['lam']

io.savemat('spatial_footprints.mat', {'array': spatial_footprints})
# %%
