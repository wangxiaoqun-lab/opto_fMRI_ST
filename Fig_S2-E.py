import os
import numpy as np
import pandas as pd
import glob
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from scipy import stats
from nilearn import plotting,image,masking,signal
from PIL import Image
from sklearn.decomposition import PCA
from matplotlib import gridspec
import matplotlib as mpl
import cairosvg

mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False

try:
    plt.rcParams['font.family'] = 'Arial'
except Exception as e:
    plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams.update(plt.rcParamsDefault)
plt.rcParams['font.size'] = 12
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12


from Region_dict import CCF_new




input_motions = ...# a list of path to the motion file
input_inject_masks = ...# a list of path to the inject site mask file
input_3dvolreg = ...# a list of path to the 3dvolreg file
input_muscle_imgs = ...# a list of path to the muscle mask image
input_list=...# a list of the name of session,run,run_prefix

save_path_figure = ...# path the save the figure

#template files
DSURQE_dir=... #dir to store DSURQE images
reference_file=...#CCF template in DSURQE space
atlas_img=...#CCF atlas in DSURQE space
bg_img_mask = ...#DSURQE mask


bg_img=image.load_img(reference_file)
bg_img_atlas = image.load_img(atlas_img)
bg_img_atlas_data = np.array(bg_img_atlas.dataobj)
bg_img_atlas_data[bg_img_atlas_data > 0] = 1
bg_img_mask = image.resample_to_img(image.new_img_like(bg_img_atlas, bg_img_atlas_data), bg_img, interpolation='nearest')
bg_img_brain = masking.unmask(masking.apply_mask(bg_img, bg_img_mask), bg_img_mask)



def svg_to_png(svg_file, png_file):
    cairosvg.svg2png(url=svg_file, write_to=png_file)

def merge_svgs_to_png(svg_files, output_file, horizontal=True, spacing=10):
    png_images = []
    
    for i, svg_file in enumerate(svg_files):
        png_file = f"temp_image_{i}.png"
        svg_to_png(svg_file, png_file)
        png_images.append(Image.open(png_file))

    if horizontal:
        total_width = sum(img.width for img in png_images) + spacing * (len(png_images) - 1)
        max_height = max(img.height for img in png_images)
        merged_image = Image.new('RGBA', (total_width, max_height), (255, 255, 255, 0))
    else:
        max_width = max(img.width for img in png_images)
        total_height = sum(img.height for img in png_images) + spacing * (len(png_images) - 1)
        merged_image = Image.new('RGBA', (max_width, total_height), (255, 255, 255, 0))

    offset = 0
    for img in png_images:
        if horizontal:
            merged_image.paste(img, (offset, 0))
            offset += img.width + spacing
        else:
            merged_image.paste(img, (0, offset))
            offset += img.height + spacing

    merged_image.save(output_file)

def calculate_FD(motion_params):
    """calculate framewise displacement (FD) as per Power et al., 2012
    edited basd on https://github.com/FCP-INDI/C-PAC/blob/master/CPAC/generate_motion_statistics/generate_motion_statistics.py
    
    Args:
        motion_params (1-d array): matrix of motion parameters (roll, pitch, yaw, dS, dL, dP)
    Returns:
        str: framewise displacement
    """
    motion_params = motion_params.T
    rotations = np.transpose(np.abs(np.diff(motion_params[:3 :])))
    translations = np.transpose(np.abs(np.diff(motion_params[3:6, :])))

    fd = np.sum(translations, axis=1) + \
         (5 * np.pi / 180) * np.sum(rotations, axis=1)
    # mice 5mm radius
    fd = np.insert(fd, 0, 0)
    return fd


def find_unique_file(folder_paths, file_pattern):
    matching_files = []
    for folder_path in folder_paths:
        file_path = glob.glob(os.path.join(folder_path, file_pattern))
        if len(file_path) > 1:
            raise ValueError(f"Multiple files found for pattern '{os.path.join(folder_path, file_pattern)}' in folder ")
        elif len(file_path) == 1:
            matching_files.append(os.path.abspath(file_path[0]))

    if len(matching_files) == 0:
        raise ValueError(f"No files found for pattern '{os.path.join(folder_path, file_pattern)}' in any folder")
    return matching_files[0]


def motion_derivatives(confounds):
    # translation/rotation + derivatives (12 parameters) + Scrubbing
    diff_confounds = np.around(confounds - np.vstack((confounds[0, :], confounds[:-1, :])), decimals=6)
    confounds = np.hstack((confounds, diff_confounds))

    return confounds

def cal_DVARS(timeseries):
    derivative=np.concatenate((np.empty([1,timeseries.shape[1]]),timeseries[1:,:]-timeseries[:-1,:]))
    DVARS=np.sqrt((derivative**2).mean(axis=1))
    return DVARS

from scipy.stats import gamma 



def draw_combine3(img_3dvolreg_DSURQE, bg_img_mask,fd,inject_mask,prefix,save_path=None,axes=None,title="No regression",fig_width=15):

    mpl.rcParams.update(mpl.rcParamsDefault)
    bg_img_mask_res = image.resample_to_img(bg_img_mask, img_3dvolreg_DSURQE, interpolation='nearest')
    brain_timeseries = masking.apply_mask(img_3dvolreg_DSURQE, bg_img_mask_res)

    inject_timeseries = np.mean(masking.apply_mask(img_3dvolreg_DSURQE, inject_mask),axis=1)
    inject_timeseries =signal.clean(inject_timeseries.reshape(len(inject_timeseries), 1), detrend=True, standardize='zscore', low_pass=0.2, high_pass=0.01, t_r=1).flatten()

    DVARS=cal_DVARS(brain_timeseries)
    global_signal = np.mean(brain_timeseries,axis=1)
    
    if axes is None:
        gs = gridspec.GridSpec(4, 1, height_ratios=[1, 1,1, 6])  
        fig = plt.figure(figsize=(fig_width, 8)) #6
        plt.subplots_adjust(bottom=0.1,top=0.9)

    #create subplots via GridSpec
    ax0 = fig.add_subplot(gs[0, 0])  
    ax1 = fig.add_subplot(gs[1, 0])  
    ax2 = fig.add_subplot(gs[2, 0])  
    ax3 = fig.add_subplot(gs[3, 0])  

    
    ax0.plot(global_signal[1:]/1000,label="gs", color='gold')
    ax0.legend(['gs'], fontsize=8, loc='upper left', bbox_to_anchor=(-0.12, 0.5))
    ax0.text(1,1.05,'gs vs FD: r = ' + str(stats.pearsonr(fd[1:], global_signal[1:])), verticalalignment='bottom', horizontalalignment='right', transform=ax0.transAxes)
    ax0.text(0,1.05, prefix , verticalalignment='bottom', horizontalalignment='left', transform=ax0.transAxes)

    ax0_ = ax0.twinx()
    ax0_.plot(fd[1:],label="FD", color='grey')
    ax0_.legend(['FD'], fontsize=8, loc='upper left', bbox_to_anchor=(-0.12, 1.05))
    ax0_.set_xlim(0, fd[1:].shape[-1])
    ax0_.set_xlim(0, global_signal[1:].shape[-1])
    
    
    ax1.plot(fd[1:],label="FD", color='grey')
    ax1.legend(['FD'], fontsize=8, loc='upper left', bbox_to_anchor=(-0.12, 0.5))
    ax1.set_xlim(0, global_signal[1:].shape[-1])
    
    ax1_ = ax1.twinx()
    ax1_.plot(DVARS[1:],label="DVARS",color="green")
    ax1_.text(1,1.05,'DVAS vs FD: r = ' + str(stats.pearsonr(fd[1:], DVARS[1:])), verticalalignment='bottom', horizontalalignment='right', transform=ax1.transAxes)
    ax1_.legend(['DVARS'], fontsize=8, loc='upper left', bbox_to_anchor=(-0.12, 1.05))
    ax1_.set_xlim(0, DVARS[1:].shape[-1])


    ax2.plot(inject_timeseries[1:],label="inject ts", color='k')
    ax2.legend(['inject ts'], fontsize=8, loc='upper left', bbox_to_anchor=(-0.12, 0.5))
    ax2.set_xlim(0, inject_timeseries[1:].shape[-1])
    ax2.text(1,1.05,'injection roi ts vs hrf ', verticalalignment='bottom', horizontalalignment='right', transform=ax2.transAxes)

    ax3.set_xticks([])
    ax3.set_yticks([])

    display=plotting.plot_carpet(img_3dvolreg_DSURQE,mask_img=bg_img_mask_res,axes=ax3,vmax=1,vmin=-1)
    x_label = np.append(np.array([0]),np.arange(10, 310, 60))
    ax3.xaxis.set_ticks(x_label)
    ax3.xaxis.set_ticklabels([str(tick) for tick in x_label])
    img = display.axes[3].get_images()

    cbar = plt.colorbar(img[0], ax=ax3, orientation='horizontal',fraction=0.05,aspect=5, pad=0.1,anchor=(1,0),location="bottom",label="BOLD(z score)")
    ax3.set_ylabel('{} bold signal \n voxel'.format(title),fontsize=15, labelpad=30, loc='center')
    
    for ax in [ax0,ax0_,ax1,ax1_,ax2]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_position(('outward', 10))
        ax.spines['bottom'].set_visible(False)
        
        ax.set_xticks([])

    if save_path is not None:
        plt.savefig(save_path,dpi=300)
        
        
def draw_combine4(img_3dvolreg_DSURQE, bg_img_mask,fd,head_motion,prefix,save_path=None,axes=None,title="No regression",fig_width=15):

    mpl.rcParams.update(mpl.rcParamsDefault)
    bg_img_mask_res = image.resample_to_img(bg_img_mask, img_3dvolreg_DSURQE, interpolation='nearest')
    brain_timeseries = masking.apply_mask(img_3dvolreg_DSURQE, bg_img_mask_res)

    DVARS=cal_DVARS(brain_timeseries)
    global_signal = np.mean(brain_timeseries,axis=1)
    
    if axes is None:
        gs = gridspec.GridSpec(5, 1, height_ratios=[1, 1,1,1, 5.5]) 
        fig = plt.figure(figsize=(fig_width, 10))
        plt.subplots_adjust(bottom=0.1,top=0.9)

    
    # create subplots via GridSpec
    ax0 = fig.add_subplot(gs[0, 0]) 
    ax1 = fig.add_subplot(gs[1, 0])  
    ax2 = fig.add_subplot(gs[2, 0])  
    ax3 = fig.add_subplot(gs[3, 0])
    ax4 = fig.add_subplot(gs[4, 0])

    
    ax0.plot(global_signal[1:]/1000,label="gs", color='gold')
    ax0.legend(['gs'], fontsize=8, loc='upper left', bbox_to_anchor=(-0.12, 0.5))
    ax0.text(1,1.05,'Corr with FD: r = ' + str(stats.pearsonr(fd[1:], global_signal[1:])), verticalalignment='bottom', horizontalalignment='right', transform=ax0.transAxes)
    ax0.text(0,1.05, prefix , verticalalignment='bottom', horizontalalignment='left', transform=ax0.transAxes)

    ax0_ = ax0.twinx()
    ax0_.plot(fd[1:],label="FD", color='grey')
    ax0_.legend(['FD'], fontsize=8, loc='upper left', bbox_to_anchor=(-0.12, 1.05))
    
    
    ax1.plot(fd[1:],label="FD", color='grey')
    ax1.legend(['FD'], fontsize=8, loc='upper left', bbox_to_anchor=(-0.12, 0.5))
    
    ax1_ = ax1.twinx()
    ax1_.plot(DVARS[1:],label="DVARS",color="green")
    ax1_.text(1,1.05,'Corr with FD: r = ' + str(stats.pearsonr(fd[1:], DVARS[1:])), verticalalignment='bottom', horizontalalignment='right', transform=ax1.transAxes)
    ax1_.legend(['DVARS'], fontsize=8, loc='upper left', bbox_to_anchor=(-0.12, 1.05))


    ax2.plot(head_motion[:, :3])
    ax2.text(1,1.05,'head motion -- ratations ', verticalalignment='bottom', horizontalalignment='right', transform=ax2.transAxes)
    ax2.legend(['roll', 'pitch', 'yaw'], fontsize=8, loc='upper left', bbox_to_anchor=(-0.12, 1.05))

    ax3.plot(head_motion[:, -3:])
    ax3.text(1,1.05,'head motion -- translation ', verticalalignment='bottom', horizontalalignment='right', transform=ax3.transAxes)
    ax3.legend(['dS', 'dL', 'dP'], fontsize=8, loc='upper left', bbox_to_anchor=(-0.12, 1.05))

    
    display=plotting.plot_carpet(img_3dvolreg_DSURQE,mask_img=bg_img_mask_res,axes=ax4,vmax=1,vmin=-1)
    x_label = np.append(np.array([0]),np.arange(10, 310, 60))
    ax4.xaxis.set_ticks(x_label)
    ax4.xaxis.set_ticklabels([str(tick) for tick in x_label])
    img = display.axes[4].get_images()

    # set color bar
    cbar = plt.colorbar(img[0], ax=ax4, orientation='horizontal',fraction=0.05,aspect=5, pad=0.1,anchor=(1,0),location="bottom",label="BOLD(z score)")
    ax4.set_ylabel('{} bold signal \n voxel'.format(title),fontsize=15, labelpad=30, loc='center')

    
    for ax in [ax0,ax0_,ax1,ax1_,ax2,ax3]:
        # remove the background and border of the axis
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_position(('outward', 10))
        ax.spines['bottom'].set_visible(False)
        ax.set_xlim(0, fd[1:].shape[-1])
        ax.set_xticks([])

    if save_path is not None:
        plt.savefig(save_path,dpi=300)

def combine_images(image_paths, output_path):

    images = [Image.open(path) for path in image_paths]

    # calculate the size of the new image
    width = max(img.size[0] for img in images)
    height = sum(img.size[1] for img in images)
    new_img = Image.new('RGB', (width, height), (255, 255, 255))

    # concatenate images
    y_offset = 0
    for img in images:
        new_img.paste(img, (0, y_offset))
        y_offset += img.size[1]


    new_img.save(output_path)
    return new_img



def draw_DVAS2(inputs_all,fig_width=15,pca_component=30):
    try:
        input_, input_3dvolreg_,muscle_img_, input_inject_mask_, input_motion_ = inputs_all
        mouse, session, prefix = input_

        print("input_motion_: ",input_motion_)
        print("input_3dvolreg_: ",input_3dvolreg_)
        print("muscle_img: ",muscle_img_)
        print("input_inject_mask_: ",input_inject_mask_)

        

        final_png = os.path.join(save_path_figure, mouse, 'summary_' + mouse + '_' + session + '_' + prefix  + '.png')
        if os.path.exists(final_png):
            return

        save_path_ = os.path.join(save_path_figure, mouse)
        save_path_svg_ = os.path.join(save_path_figure,"svg", mouse)
        if not os.path.exists(save_path_):
            os.makedirs(save_path_,exist_ok=True)
        if not os.path.exists(save_path_svg_):
            os.makedirs(save_path_svg_,exist_ok=True)
        save_path_svg_ = os.path.join(save_path_figure,"svg")
        
        # calculate fd
        input_motion_ = os.path.join(os.path.dirname(input_3dvolreg_),"head_motion.txt")
        head_motion = pd.DataFrame(np.loadtxt(input_motion_), columns = 'roll|pitch|yaw|dS|dL|dP'.split('|'))
        
    
        #resolution aglinment
        img_3dvolreg_DSURQE = image.load_img(input_3dvolreg_) 
        img_3dvolreg_DSURQE = image.clean_img(img_3dvolreg_DSURQE, detrend=True,standardize=False,
                                       t_r=1,confounds=None)

        bg_img_mask_ = image.load_img(bg_img_mask)

        if img_3dvolreg_DSURQE.shape[-1]>300:
            start_index = img_3dvolreg_DSURQE.shape[-1]-310
            img_3dvolreg_DSURQE = image.index_img(img_3dvolreg_DSURQE,np.arange(start_index,img_3dvolreg_DSURQE.shape[-1]))
            head_motion = pd.DataFrame(np.loadtxt(input_motion_)[start_index:, :], columns = 'roll|pitch|yaw|dS|dL|dP'.split('|'))
            
        else:
            print("abnormal length")
            return 
        
        fd = calculate_FD(head_motion.values)

    
        save_path = os.path.join(save_path_svg_, mouse, 'fMRI5_' + mouse + '_' + session + '_' + prefix  + '.svg')
        draw_combine4(img_3dvolreg_DSURQE, bg_img_mask,fd,head_motion.values,prefix,save_path=save_path,title="No regression",fig_width=fig_width)

        save_path = os.path.join(save_path_svg_, mouse, 'fMRI3_' + mouse + '_' + session + '_' + prefix  + '.svg')


        inject_img = image.load_img(input_inject_mask_)
        inject_mask_img_res = image.resample_to_img(inject_img, img_3dvolreg_DSURQE, interpolation='nearest')

        #muscle_image
        muscle_img = image.load_img(muscle_img_)
        img_3dvolreg_sm = image.smooth_img(img_3dvolreg_DSURQE, 0.45)
        img_mask_res_ = image.resample_to_img(muscle_img, img_3dvolreg_DSURQE, interpolation='nearest')
    
        muscle_timeseries = masking.apply_mask(img_3dvolreg_sm, img_mask_res_)
        pca = PCA(n_components=pca_component)
        pca10=pca.fit_transform(muscle_timeseries)#n_time * 10
        print(pca.explained_variance_ratio_.sum())
    
        #confounds
        motion_confounds=motion_derivatives(head_motion.values)
        print(motion_confounds.shape)
        motion_confounds_=np.hstack((motion_confounds, pca10))

        img_3dvolreg_clean_DSURQE=image.clean_img(img_3dvolreg_sm, detrend=False,standardize=False,
                                       t_r=1,confounds=motion_confounds)

        draw_combine3(img_3dvolreg_clean_DSURQE, bg_img_mask,fd,inject_mask_img_res,stats.zscore(design_matrix.values[:, 0]),prefix,save_path,title="12 regressors",fig_width=fig_width)

        img_3dvolreg_clean_DSURQE=image.clean_img(img_3dvolreg_sm, detrend=False,standardize=False,
                                       t_r=1,confounds=motion_confounds_)
        save_path = os.path.join(save_path_svg_, mouse, 'fMRI2_' + mouse + '_' + session + '_' + prefix  + '.svg')

        draw_combine3(img_3dvolreg_clean_DSURQE, bg_img_mask,fd,inject_mask_img_res,stats.zscore(design_matrix.values[:, 0]),prefix,save_path,title="6+PCAs regressors",fig_width=fig_width)


    
        output_file=os.path.join(save_path_figure, mouse, 'summary_' + prefix  + '.png')
        svg_files = [
                os.path.join(save_path_svg_, mouse, 'fMRI5_' + mouse + '_' + session + '_' + prefix  + '.svg'),
                os.path.join(save_path_svg_, mouse, 'fMRI3_' + mouse + '_' + session + '_' + prefix  + '.svg'),
                os.path.join(save_path_svg_, mouse, 'fMRI2_' + mouse + '_' + session + '_' + prefix  + '.svg')
            ]
        merge_svgs_to_png(svg_files, output_file, horizontal=False, spacing=-70)
        print("saving...:",os.path.join(save_path_figure, mouse, 'summary_' + prefix  + '.png'))  
    except Exception as e:
        print("error: ",e)




inputs_alls = []
 
for a, b, c,f, d in zip(input_list,  input_3dvolreg,input_muscle_imgs, input_inject_masks, input_motions):
    inputs_alls.append([a, b, c,f, d])

print("inputs_alls: ",len(inputs_alls))
os.makedirs(save_path_figure,exist_ok=True)
for inputs_all in inputs_alls:
    draw_DVAS2(inputs_all,fig_width=15,pca_component=30)


