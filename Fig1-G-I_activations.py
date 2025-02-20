from nilearn import image,plotting,masking
import matplotlib.pyplot as plt
import numpy as np
from utils import levels,contrast_cmap

pl_ccf_tmap=... #path to PL tmap
sub_ccf_tmap=... #path to Sub tmap 
pl_proj_map=... #path to PL projection map 
sub_proj_map=... #path to Sub projection tmap 
pl_cfos_tmap=... #path to PL cfos map
sub_cfos_tmap=... #path to Sub cfos map 

#template files
Allen_dir=... #dir to store allen images
reference_file=Allen_dir+"P56_Atlas.nii.gz"
contour_img=Allen_dir+"P56_Annotation.nii.gz"


ccf_img = image.load_img(contour_img)
ccf_bg_img = masking.compute_background_mask(ccf_img)
#mask out the noise of outside of the Allen Atlas brain for bg img
new_bg_img=masking.unmask(masking.apply_mask(reference_file,ccf_bg_img),ccf_bg_img)



def mask_brain_out(stat_img,mask_img):
    mask_img = image.resample_to_img(mask_img,stat_img,interpolation='nearest')
    stat_masked_img = masking.apply_mask(stat_img,mask_img)
    stat_masked_img = masking.unmask(stat_masked_img,mask_img)
    return stat_masked_img



#convert allen atlas slice coordination into plot ax coordination
slice_index = [382,347,312,275,232,196,166]
bg_img = image.load_img(reference_file)
affine_ccf = bg_img.affine
y_coor = [image.coord_transform(0, y_, 0, affine_ccf)[1] for y_ in slice_index]


#####################################draw mPFC fmri activation map#######################################
fig = plt.figure(figsize=(10, 0.9),dpi=500)

PL_group_ofMRI=plotting.plot_stat_map(mask_brain_out(pl_ccf_tmap, ccf_bg_img), display_mode='y', 
                               cut_coords=y_coor, symmetric_cbar=False, cmap=contrast_cmap(50),
                               threshold=0, bg_img=new_bg_img, annotate=False,
                               black_bg=False, figure=fig)
colorbar = PL_group_ofMRI._cbar
colorbar.set_ticks(np.linspace(np.int32(0),np.int32(25),num=4,dtype=np.int32))
PL_group_ofMRI.add_contours(
                contour_img,colors="dimgray",linewidths=0.5,levels=levels[::5])
print("saving fig1-G_PL.png")
fig.savefig("fig1-G_PL.png",dpi=500)



#draw Sub fmri activation map
fig = plt.figure(figsize=(10, 0.9),dpi=500)
sub_group_ofMRI  = plotting.plot_stat_map(mask_brain_out(sub_ccf_tmap, ccf_bg_img), display_mode='y', cmap=contrast_cmap(80),
                               cut_coords=y_coor, symmetric_cbar=False,
                               threshold=0, bg_img=new_bg_img, annotate=False,
                               black_bg=False, figure=fig)
colorbar = sub_group_ofMRI._cbar
colorbar.set_ticks(np.linspace(np.int32(0),np.int32(17),num=4,dtype=np.int32))
sub_group_ofMRI.add_contours(
                contour_img,colors="dimgray",linewidths=0.5,levels=levels[::5])
print("saving fig1-G_Sub.png")
fig.savefig("fig1-G_Sub.png",dpi=500)


####################################draw mPFC log projection map####################################
pl_proj_map_log = image.math_img("np.where(np.log(img1)<-5.5,np.nan,np.log(img1))",img1=pl_proj_map)
vmin=None
fig = plt.figure(figsize=(9, 0.8),dpi=500)
display_mode="y"
disp  = plotting.plot_stat_map(pl_proj_map_log, display_mode=display_mode, 
                               cut_coords=y_coor, symmetric_cbar=False,
                               bg_img=new_bg_img,annotate=False,
                               cmap="cold_white_hot_r",
                               black_bg=False,figure=fig)
colorbar = disp._cbar
colorbar.set_ticks(np.linspace(-6,0,num=4,dtype=np.int32))
disp.add_contours(
    contour_img,colors="dimgray",linewidths=0.5,levels=levels[::5])
print("saving fig1-H_PL.png")
fig.savefig("fig1-H_PL.png",dpi=500)


#draw Sub log projection map
sub_proj_map_log = image.math_img("np.where(np.log(img1)<-5.5,np.nan,np.log(img1))",img1=sub_proj_map)
fig = plt.figure(figsize=(9, 0.8),dpi=500)
display_mode="y"
disp  = plotting.plot_stat_map(sub_proj_map_log, display_mode=display_mode, 
                               cut_coords=y_coor, symmetric_cbar=False,
                               bg_img=new_bg_img,annotate=False,
                               cmap="cold_white_hot_r",
                               black_bg=False,figure=fig)
colorbar = disp._cbar
colorbar.set_ticks(np.linspace(-6,0,num=4,dtype=np.int32))
disp.add_contours(
    contour_img,colors="dimgray",linewidths=0.5,levels=levels[::5])
print("saving fig1-H_Sub.png")
fig.savefig("fig1-H_Sub.png",dpi=500)

############################################draw mPFC cfos tmap###########################################
fig = plt.figure(figsize=(10, 0.9),dpi=500)
display_mode="y"
PL_group_  = plotting.plot_stat_map(pl_cfos_tmap, display_mode=display_mode, cut_coords=y_coor, symmetric_cbar=False,
                               threshold=2.19, bg_img=new_bg_img, annotate=False,black_bg=False, figure=fig)
colorbar = PL_group_._cbar
colorbar.set_ticks(np.linspace(np.int32(0),np.int32(11),num=3,dtype=np.int32))
PL_group_.add_contours(
                contour_img,colors="dimgray",linewidths=0.5,levels=levels[::5])
print("saving fig1-I_PL.png")
fig.savefig("fig1-I_PL.png",dpi=500)


#draw Sub cfos tmap
fig = plt.figure(figsize=(10, 0.9),dpi=500)
display_mode="y"
sub_group_  = plotting.plot_stat_map(sub_cfos_tmap, display_mode=display_mode, cut_coords=y_coor, symmetric_cbar=False,
                               threshold=3.18, bg_img=new_bg_img, annotate=False,black_bg=False, figure=fig)
colorbar = sub_group_._cbar
colorbar.set_ticks(np.linspace(np.int32(0),np.int32(7),num=3,dtype=np.int32))
sub_group_.add_contours(
                contour_img,colors="dimgray",linewidths=0.5,levels=levels[::5])
print("saving fig1-I_Sub.png")
fig.savefig("fig1-I_Sub.png",dpi=500)