from Region_dict import All_regions, Thalamus
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import os
from nilearn import signal

ts_dir=... #path to the timeseries folder
PL_runs_test=... #a list of path of the PL runs test in template space
PL_runs_control=... #a list of path of the PL runs control in template space    
Sub_runs_test=... #a list of path of the Sub runs test in template space
Sub_runs_control=... #a list of path of the Sub runs control in template space

def cal_ts_mean(ts):
    for i in range(ts.shape[0]):
        ts[i, :] = ts[i, :] / np.mean(ts[i, 10:30]) 
    return ts


ts_PL = []
for run in PL_runs_test:
    run_name = run.split("/")[-1].replace("_62regre_in_DSURQE.nii.gz","")
    mouse_id = run.split("/")[-2]
    if os.path.exists(os.path.join(ts_dir, mouse_id, run_name + '_ts_atlas.npy')):
        ts_ = np.load(os.path.join(ts_dir, mouse_id, run_name + '_ts_atlas.npy'))
        if len(ts_[ts_==0]) == 0:
            if ts_.shape[1] == 330:
                ts_ = cal_ts_mean(ts_)
                ts_f = signal.clean(ts_.T, detrend=True, standardize=False,low_pass=0.2, high_pass=0.01, t_r=1).T
                ts_PL.append(ts_f)

            elif ts_.shape[1] == 340:
                ts_ = cal_ts_mean(ts_[:, 10:])
                ts_f = signal.clean(ts_.T, detrend=True, standardize=False,low_pass=0.2, high_pass=0.01, t_r=1).T
                ts_PL.append(ts_f)

print("len(PL_runs_test),len(ts_PL): ",len(PL_runs_test),len(ts_PL))

        
ts_PL_con = []
for run in PL_runs_test:
    run_name = run.split("/")[-1].replace("_62regre_in_DSURQE.nii.gz","")
    mouse_id = run.split("/")[-2]
    if os.path.exists(os.path.join(ts_dir, mouse_id, run_name + '_ts_atlas.npy')):
        ts_ = np.load(os.path.join(ts_dir, mouse_id, run_name + '_ts_atlas.npy'))
        if len(ts_[ts_==0]) == 0:
            if ts_.shape[1] == 330:
                ts_ = cal_ts_mean(ts_)
                ts_f = signal.clean(ts_.T, detrend=True, standardize=False,low_pass=0.2, high_pass=0.01, t_r=1).T
                ts_PL_con.append(ts_f)

            elif ts_.shape[1] == 340:
                ts_ = cal_ts_mean(ts_[:, 10:])
                ts_f = signal.clean(ts_.T, detrend=True, standardize=False,low_pass=0.2, high_pass=0.01, t_r=1).T
                ts_PL_con.append(ts_f)
                
print("len(ts_PL_con),len(PL_runs_control): ",len(ts_PL_con),len(PL_runs_control))

ts_PL_con_trials=[]
for pl_ts in ts_PL_con:
    for i in range(5):
        mice_ts_trial = pl_ts[:,10+60*i:10+80+60*i]
        ts_PL_con_trials.append(mice_ts_trial)

ts_PL_con_mean=np.nanmean(np.array(ts_PL_con_trials),axis=0)

print("ts_PL_con_mean.shape: ",ts_PL_con_mean.shape)

ts_PL_trials=[]
for pl_ts in ts_PL:
    for i in range(5):
        mice_ts_trial = pl_ts[:,10+60*i:10+80+60*i]-ts_PL_con_mean
        ts_PL_trials.append(mice_ts_trial)
        
stimulation_points = list(np.array([20]))
stimulation_length = 20

n_rows = int(np.ceil(len(Thalamus) / 3))


fig, axes = plt.subplots(n_rows, 4, figsize=(20, 2.5*n_rows))

#each row has 4 subplots
if len(Thalamus) % 4 != 0:
    axes = axes.ravel()

for i, region in enumerate(Thalamus):
    ax = axes[i]
    
    for start in stimulation_points:
        ax.axvspan(start, start + stimulation_length, color='grey', alpha=0.15)  
        
    line1, = ax.plot(np.nanmean(ts_PL_trials, axis=0)[All_regions.index(region)*2+1, :], color='k', label='right')
    line2, = ax.plot(np.nanmean(ts_PL_trials, axis=0)[All_regions.index(region)*2, :], color='#cccccc', label='left')

    tmp_ts = np.nanmean(ts_PL_trials, axis=0)[All_regions.index(region)*2, :]
    max_idx = np.argmax(tmp_ts[20:40])+20
    ax.axhline(y=tmp_ts[max_idx], xmax=max_idx / len(tmp_ts), color='b', linestyle='--')
    ymin, ymax = ax.get_ylim()
    ax.axvline(x=max_idx,ymax=tmp_ts[max_idx]+0.01, color='b')
    ax.plot([max_idx, max_idx], [ymin, tmp_ts[max_idx]], '--', color='b')

    text_y=ymin-(ymax - ymin) * 0.02

    
    ax.text(x=-4, y=tmp_ts[max_idx] + (ymax - ymin) * 0.08, s=f'{tmp_ts[max_idx]*100:.2f}%', color='b', verticalalignment='center', horizontalalignment='right')
    ax.text(x=max_idx, y=text_y, s=f'{max_idx}', color='b', verticalalignment='center', horizontalalignment='right')


    tmp_ts = np.nanmean(ts_PL_trials, axis=0)[All_regions.index(region)*2+1, :]
    max_idx = np.argmax(tmp_ts)
    ax.axhline(y=tmp_ts[max_idx], xmax=max_idx / len(tmp_ts), color='r', linestyle='--')
    ymin, ymax = ax.get_ylim()
    ax.plot([max_idx, max_idx], [ymin, tmp_ts[max_idx]], 'r--')
    ax.axvline(x=max_idx,ymax=tmp_ts[max_idx]+0.01, color='r')

    
    ax.text(x=-4, y=tmp_ts[max_idx] + (ymax - ymin) * 0.08, s=f'{tmp_ts[max_idx]*100:.2f}%', color='r', verticalalignment='center', horizontalalignment='right')
    ax.text(x=max_idx, y=text_y, s=f'{max_idx}', color='r', verticalalignment='center', horizontalalignment='right')

    # set title, xlabel, ylabel
    ax.set_title(region)
    ax.set_xlabel('time')
    ax.set_ylabel('BOLD change')
    
    # convert y to percent
    def to_percent(y, position):
        return "{:.1f}%".format(100 * y)

    # transform y to percent
    formatter = FuncFormatter(to_percent)
    ax.yaxis.set_major_formatter(formatter)

    # add legend
    ax.legend(handles=[line1, line2], loc='upper right', frameon=False, fontsize=9)

    # remove the top and right border
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
# close redundant subplots
if len(Thalamus) % 4 != 0:
    for j in range(len(Thalamus), n_rows*4):
        fig.delaxes(axes[j])

plt.tight_layout()
print("saving ./images/thalamus_PL.svg")
plt.savefig('thalamus_PL.svg', dpi=300)
plt.show()
        
        
        
ts_Sub = []
for run in Sub_runs_test:
    run_name = run.split("/")[-1].replace("_42regre_in_DSURQE.nii.gz","")
    mouse_id = run.split("/")[-2]
    if os.path.exists(os.path.join(ts_dir, mouse_id, run_name + '_ts_atlas.npy')):
        ts_ = np.load(os.path.join(ts_dir, mouse_id, run_name + '_ts_atlas.npy'))
        if len(ts_[ts_==0]) == 0:
            if ts_.shape[1] == 330:
                ts_ = cal_ts_mean(ts_)
                ts_f = signal.clean(ts_.T, detrend=True, standardize=False,low_pass=0.2, high_pass=0.01, t_r=1).T
                ts_Sub.append(ts_f)

            elif ts_.shape[1] == 340:
                ts_ = cal_ts_mean(ts_[:, 10:])
                ts_f = signal.clean(ts_.T, detrend=True, standardize=False,low_pass=0.2, high_pass=0.01, t_r=1).T
                ts_Sub.append(ts_f)       

print("len(ts_Sub),len(Sub_runs_test): ",len(ts_Sub),len(Sub_runs_test))

ts_Sub_con = []
for run in Sub_runs_control:
    run_name = run.split("/")[-1].replace("_22regre_in_DSURQE.nii.gz","")
    mouse_id = run.split("/")[-2]
    if os.path.exists(os.path.join(ts_dir, mouse_id, run_name + '_ts_atlas.npy')):
        ts_ = np.load(os.path.join(ts_dir, mouse_id, run_name + '_ts_atlas.npy'))
        if len(ts_[ts_==0]) == 0:
            if ts_.shape[1] == 330:
                ts_ = cal_ts_mean(ts_)
                ts_f = signal.clean(ts_.T, detrend=True, standardize=False,low_pass=0.2, high_pass=0.01, t_r=1).T
                ts_Sub_con.append(ts_f)

            elif ts_.shape[1] == 340:
                ts_ = cal_ts_mean(ts_[:, 10:])
                ts_f = signal.clean(ts_.T, detrend=True, standardize=False,low_pass=0.2, high_pass=0.01, t_r=1).T
                ts_Sub_con.append(ts_f)        
        
print("len(ts_Sub_con),len(Sub_runs_control): ",len(ts_Sub_con),len(Sub_runs_control))   

ts_Sub_con_trials=[]
for pl_ts in ts_Sub_con:
    for i in range(5):
        mice_ts_trial = pl_ts[:,10+60*i:10+80+60*i]
        ts_Sub_con_trials.append(mice_ts_trial)
        
ts_PL_con_mean=np.nanmean(np.array(ts_Sub_con_trials),axis=0)
print("ts_PL_con_mean.shape: ",ts_PL_con_mean.shape)

ts_Sub_trials=[]
for pl_ts in ts_Sub:
    for i in range(5):
        mice_ts_trial = pl_ts[:,10+60*i:10+80+60*i]-ts_PL_con_mean
        ts_Sub_trials.append(mice_ts_trial)

stimulation_points = list(np.array([20]))
stimulation_length = 20

n_rows = int(np.ceil(len(Thalamus) / 3))

fig, axes = plt.subplots(n_rows, 4, figsize=(20, 2.5*n_rows))

if len(Thalamus) % 4 != 0:
    axes = axes.ravel()

for i, region in enumerate(Thalamus):
    ax = axes[i]
    
    for start in stimulation_points:
        ax.axvspan(start, start + stimulation_length, color='grey', alpha=0.15)  
        
    line1, = ax.plot(np.nanmean(ts_Sub_trials, axis=0)[All_regions.index(region)*2+1, :], color='k', label='right')
    line2, = ax.plot(np.nanmean(ts_Sub_trials, axis=0)[All_regions.index(region)*2, :], color='#cccccc', label='left')

    tmp_ts = np.nanmean(ts_Sub_trials, axis=0)[All_regions.index(region)*2, :]
    max_idx = np.argmax(tmp_ts[20:40])+20
    ax.axhline(y=tmp_ts[max_idx], xmax=max_idx / len(tmp_ts), color='b', linestyle='--')
    ymin, ymax = ax.get_ylim()
    ax.axvline(x=max_idx,ymax=tmp_ts[max_idx]+0.01, color='b')
    ax.plot([max_idx, max_idx], [ymin, tmp_ts[max_idx]], '--', color='b')

    text_y=ymin-(ymax - ymin) * 0.02

    
    ax.text(x=-4, y=tmp_ts[max_idx] + (ymax - ymin) * 0.08, s=f'{tmp_ts[max_idx]*100:.2f}%', color='b', verticalalignment='center', horizontalalignment='right')
    ax.text(x=max_idx, y=text_y, s=f'{max_idx}', color='b', verticalalignment='center', horizontalalignment='right')


    tmp_ts = np.nanmean(ts_Sub_trials, axis=0)[All_regions.index(region)*2+1, :]
    max_idx = np.argmax(tmp_ts[20:40])+20
    ax.axhline(y=tmp_ts[max_idx], xmax=max_idx / len(tmp_ts), color='r', linestyle='--')
    ymin, ymax = ax.get_ylim()
    ax.plot([max_idx, max_idx], [ymin, tmp_ts[max_idx]], 'r--')
    ax.axvline(x=max_idx,ymax=tmp_ts[max_idx]+0.01, color='r')

    
    ax.text(x=-4, y=tmp_ts[max_idx] + (ymax - ymin) * 0.08, s=f'{tmp_ts[max_idx]*100:.2f}%', color='r', verticalalignment='center', horizontalalignment='right')
    ax.text(x=max_idx, y=text_y, s=f'{max_idx}', color='r', verticalalignment='center', horizontalalignment='right')


    ax.set_title(region)
    ax.set_xlabel('time')
    ax.set_ylabel('BOLD change')

    def to_percent(y, position):
        return "{:.1f}%".format(100 * y)

    formatter = FuncFormatter(to_percent)
    ax.yaxis.set_major_formatter(formatter)

    
    ax.legend(handles=[line1, line2], loc='upper right', frameon=False, fontsize=9)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
if len(Thalamus) % 4 != 0:
    for j in range(len(Thalamus), n_rows*4):
        fig.delaxes(axes[j])

plt.tight_layout()
print("saving ./images/thalamus_Sub.svg")
plt.savefig('thalamus_Sub.svg', dpi=300)