import matplotlib.pyplot as plt
from nilearn import image,plotting
import numpy as np
import pandas as pd

from Region_dict import All_dict, CCF_new
from utils import levels

import nibabel as nib



bg_img=...# atlas contain ROI index label



CCF_new = CCF_new.reset_index().set_index('ACR_name')

def create_density_nii(density_df, CCF_new,All_dict=None,bg_img=None):
    
    density_dict = dict(zip(density_df[density_df.keys().tolist()[0]], density_df[density_df.keys().tolist()[1]]))
    

    P56_Annotation = image.load_img(bg_img)
    P56_Annotation_data = np.array(P56_Annotation.dataobj)
    density_img = np.zeros_like(P56_Annotation_data)

    #fill values
    for region, value in density_dict.items():
        region_index = CCF_new.loc[region, 'index']
        density_img[P56_Annotation_data == region_index] = value
        if region.split('_')[0] in All_dict and len(All_dict[region.split('_')[0]]) > 0:
            for region_ in All_dict[region.split('_')[0]]:
                if region_+ region[-3:]  in CCF_new.index.tolist():
                    region_index = CCF_new.loc[region_ + "_lh", 'index']
                    density_img[P56_Annotation_data == region_index] = value

                    region_index = CCF_new.loc[region_ + "_rh", 'index']
                    density_img[P56_Annotation_data == region_index] = value


    #convert to 3d nii image
    new_nii = nib.Nifti1Image(density_img, P56_Annotation.affine, P56_Annotation.header)

    return new_nii

All_regions= list(All_dict.keys())
density_df = pd.DataFrame()
density_df['index'] = [r + '_lh' for r in All_regions] 
density_df['roi_values']=[i+1 for i in range(len(density_df['index']))]
print("density_df: ",density_df)
dict_WB_nii=create_density_nii(density_df,CCF_new,All_dict,bg_img)
dict_WB_nii.to_filename("results/roi_img.nii.gz")


######draw 2D visulization
roi_cmap = plt.cm.hsv
fig = plt.figure(figsize=(15, 1))
disp = plotting.plot_roi(dict_WB_nii,display_mode='y',cut_coords=np.flip(np.arange(-5,2,0.6)),black_bg=False, annotate=False,
                         cmap=roi_cmap,bg_img=None,figure=fig)
disp.add_contours(
    bg_img,colors="dimgray",linewidths=0.9,
    levels=levels
)
print("saving ./images/Fig_S4-B.tiff")
fig.savefig("images/Fig_S4-B.tiff",dpi=300)



######generate color-label pairs
roi_num = density_df['roi_values']
global_min=density_df['roi_values'].min()
global_max = density_df['roi_values'].max()
norm = plt.Normalize(vmin=global_min, vmax=global_max)
sm = plt.cm.ScalarMappable(cmap=roi_cmap, norm=norm)
colors = sm.to_rgba(roi_num)
fig = plt.figure(figsize=(15, 10))

density_df=density_df.set_index("roi_values")
for i in range(1,len(colors)):
    value = roi_num[i]
    color = colors[i]
    
    x = (i-1) % 10  
    y = (i-1) // 10  
    plt.scatter(x, y, color=color, s=100,figure=fig)  
    roi_name = density_df.loc[int(value),'index'].split("_")[0]
    plt.text(x+0.2, y,roi_name ,color='black', ha='left', va='center',figure=fig)  

plt.xticks(range(10))
plt.yticks(range(10))
plt.axis("off")
plt.show()
print("saving ./images/Fig_S4-B2.svg")
fig.savefig("./images/Fig_S4-B2.svg",dpi=100)