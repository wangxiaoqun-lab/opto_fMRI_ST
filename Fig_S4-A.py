import glob,os
from nilearn import image,plotting,masking

from utils import y_coor

fmri_path_p="preprocessed/sub-185/ses-*/func/*/*.3dvolreg.nii.gz"

def cal_tSNR(fmri_path):
    if os.path.exists(fmri_path.replace(".nii.gz","_tSNR.nii.gz")):
        return
    print(fmri_path)
    img = image.load_img(fmri_path)
    array = img.get_fdata()[:,:,:,30:]
    mean = array.mean(axis=-1)
    std = array.std(axis=-1)
    std_image=image.new_img_like(img,std)
    
    tSNR = np.divide(mean, std)
    tSNR[np.isnan(tSNR)]=0
    tSNR_image=image.new_img_like(img,tSNR)
    
    tSNR_image.to_filename(fmri_path.replace(".nii.gz","_tSNR.nii.gz"))
    return fmri_path.replace(".nii.gz","_tSNR.nii.gz")
    



for fmri_path in glob.glob(fmri_path_p):
    tSNR_image = cal_tSNR(fmri_path)
    tSNR_image1= cal_tSNR(fmri_path.replace(".nii.gz","_in_DSURQE.nii.gz"))
    
    
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
norm = Normalize(vmin=-10, vmax=20)
cmap = plt.cm.YlGnBu_r 
mean_img= image.load_img(fmri_path_p.replace(".nii.gz","_in_DSURQE.nii.gz"))
display_mode="y"
fig = plt.figure(figsize=(10, 0.9),dpi=500)
plotting.plot_stat_map(mean_img,cmap=cmap, display_mode=display_mode,vmax=20,symmetric_cbar=False,
                               cut_coords=y_coor,bg_img=None,figure=fig)
print("saving ./images/Fig_S4-A.png")
fig.savefig("images/Fig_S4-A.png",dpi=500)