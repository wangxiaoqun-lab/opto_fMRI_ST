import os
import numpy as np
from scipy.stats import sem
import altair as alt
import pandas as pd


pl_mice_run=... #a list of path of the PL runs test in template space
Sub_mice_run=... #a list of path of the Sub runs test in template space
ts_path_dir=...#path to the ts file


def percent_change(ts):
    baseline = np.mean(ts[:10])
    percent_change = (ts - baseline) / np.abs(baseline)
    return percent_change
    


region=['Sub',"PL"]
results={}
for reg in region:
    results[f'ts_{reg}']=[]


all_run=[pl_mice_run,Sub_mice_run]
for reg,mice_runs in zip(region,all_run):
    for mice_run in mice_runs:
        mouse_id = mice_run.split("_")[0]
        try:
            ts_path=os.path.join(ts_path_dir,mouse_id,mice_run+"_injection_sphere_.npy")
            ts_ = np.load(ts_path)
            results[f'ts_{reg}'].append(ts_)
        except Exception as e:
            print(e)
            continue
        
vis_Sub = pd.DataFrame()


key='ts_Sub'
degree=sem(np.array(results[key])[:, 0, :],axis=1)
temp_df = pd.DataFrame()
temp_df['fMRI'] = np.nanmedian(np.array(results[key]), axis=0).flatten()
temp_df['ts'] = np.arange(1, 311)
temp_df['STD'] = sem(np.array(results[key]),axis=0).flatten()  
temp_df['Upper'] = temp_df['fMRI'] + temp_df['STD']
temp_df['Lower'] = temp_df['fMRI'] - temp_df['STD']

vis_Sub = pd.concat([vis_Sub, temp_df])
    


# Stimulation points and lengths
stimulation_points = list(np.array([10, 70, 130, 190, 250]))

stimulation_length = 20

# Create a DataFrame for the rectangles
rects = pd.DataFrame({
    'start': stimulation_points,
    'end': [s + stimulation_length for s in stimulation_points]
})

base_chart_Sub = alt.Chart(vis_Sub).mark_line().encode(
    alt.X('ts:Q', axis=alt.Axis(tickMinStep=20, labelAngle=0, title=''), scale=alt.Scale(domain=[0, 320],nice=False)),
    alt.Y('fMRI:Q'),
    color=alt.value('#0020F5')
)

hrf_chart_Sub = alt.Chart(vis_Sub).mark_line().encode(
    alt.X('ts:Q', axis=alt.Axis(tickMinStep=20, labelAngle=0, title=''), scale=alt.Scale(domain=[0, 320],nice=False)),
    alt.Y('hrf:Q'),
)

# Overlay rectangles
rect_chart = alt.Chart(rects).mark_rect(color='#0020F5', opacity=0.15).encode(
    x=alt.X('start:Q', scale=alt.Scale(domain=[0, 320],nice=False)),
    x2='end:Q'
)

std_area_chart = alt.Chart(vis_Sub).mark_area(color='darkgray', opacity=0.8).encode(
    x=alt.X('ts:Q', scale=alt.Scale(domain=[0, 320],nice=False)),
    y=alt.Y('Upper:Q'),
    y2='Lower:Q'
)


# Layer the charts and apply configurations
final_chart_Sub = alt.layer(std_area_chart,base_chart_Sub, hrf_chart_Sub,rect_chart).configure_axis(
    grid=False,
    labelFontSize=12,
    titleFontSize=0,
    domainColor='#454545',
    domainWidth=1.3,
    gridColor='black',
    tickColor='#454545',
    tickWidth=1.5,
    labelOffset=0,
    labelPadding=4
).configure_view(
    strokeOpacity=0
).configure_axisX(
    offset=3
).configure_axisY(
    offset=3
).properties(
    width=500,
    height=100
)
print("saving ./images/Fig_S1-E_Sub.html")   
final_chart_Sub.save("images/Fig_S1-E_Sub.html")


vis_PL = pd.DataFrame()
key = f'ts_PL'
degree=sem(np.array(results[key])[:, 0, :],axis=1)
temp_df = pd.DataFrame()
temp_df['fMRI'] = np.median(np.array(results[key])[:, 0, :], axis=0).flatten()
temp_df['ts'] = np.arange(1, 311)
temp_df['STD'] = sem(np.array(results[key])[:, 0, :],axis=0).flatten()
temp_df['Upper'] = temp_df['fMRI'] + temp_df['STD']
temp_df['Lower'] = temp_df['fMRI'] - temp_df['STD']

vis_PL = pd.concat([vis_PL, temp_df])

# Stimulation points and lengths
stimulation_points = list(np.array([10, 70, 130, 190, 250]))
stimulation_length = 20

# Create a DataFrame for the rectangles
rects = pd.DataFrame({
    'start': stimulation_points,
    'end': [s + stimulation_length for s in stimulation_points]
})

# Base chart for the PL data
base_chart_PL = alt.Chart(vis_PL).mark_line().encode(
    alt.X('ts:Q', axis=alt.Axis(tickMinStep=20, labelAngle=0, title=''), scale=alt.Scale(domain=[0, 320],nice=False)),
    alt.Y('fMRI:Q'),
    color=alt.value('#2BA02B')
)

# Overlay rectangles
rect_chart = alt.Chart(rects).mark_rect(color='#2BA02B', opacity=0.15).encode(
    x=alt.X('start:Q', scale=alt.Scale(domain=[0, 320],nice=False)),
    x2='end:Q'
)

std_area_chart = alt.Chart(vis_PL).mark_area(color='darkgray', opacity=0.8).encode(
    x=alt.X('ts:Q', scale=alt.Scale(domain=[0, 320],nice=False)),
    y=alt.Y('Upper:Q'),
    y2='Lower:Q'
)

# Layer the charts and apply configurations
final_chart_PL = alt.layer(std_area_chart, base_chart_PL, rect_chart).configure_axis(
    grid=False,
    labelFontSize=12,
    titleFontSize=0,
    domainColor='#454545',
    domainWidth=1.3,
    gridColor='black',
    tickColor='#454545',
    tickWidth=1.5,
    labelOffset=0,
    labelPadding=4
).configure_view(
    strokeOpacity=0
).configure_axisX(
    offset=3
).configure_axisY(
    offset=3
).properties(
    width=500,
    height=100
)
print("saving ./images/Fig_S1-E_PL.html")
final_chart_PL.save("images/Fig_S1-E_PL.html")