from nilearn import image,plotting,masking
import seaborn as sns
import numpy as np
import pandas as pd
import altair as alt

group_dict = {
    'PL_test':...,#dict of PL_test,key is mice_id,value is list of paths
    'Sub_test':...#dict of Sub_test,key is mice_id,value is list of paths
}
DSURQE_img=...#DSURQE template
DSURQE_mask=...#mask for DSURQE template
DSURQE_thalamus_mask = ...#thalamus mask for DSURQE template

def create_heatmap(df,vmin=0.1):
    df = df.reset_index().melt(id_vars='index')
    df.columns = ['y', 'x', 'value']

    heatmap = alt.Chart(df).mark_rect().encode(
        x=alt.X('x:O',sort=sort_col),
        y=alt.Y('y:O',sort=sort_col),
        color=alt.Color('value:Q', scale=alt.Scale(scheme='inferno',domain=[vmin,1]))
    )

    return heatmap.configure_axis(
        grid=False,
        labelFontSize=14,
        titleFontSize=0,
        domainColor='#454545',
        domainWidth=1.3,
        gridColor='black',
        tickColor='#454545',
        tickWidth=1.5,
        labelOffset=0,
        labelPadding=4,
    ).configure_view(
        strokeOpacity=0
    ).configure_axisX(offset=3).configure_axisY(offset=3).properties(
        width=400,
        height=400
    )
    

all_runs=[]
mice_ids=[]
for group_,mice_dicts in group_dict.items():
    for mice_id,mice_paths in mice_dicts.items():
        all_runs.append(image.mean_img(mice_paths))
        mice_ids.append(mice_id)
        
tstat = masking.apply_mask(all_runs, DSURQE_mask)
    
df_similarity = pd.DataFrame(np.corrcoef(tstat),columns=mice_ids,
                            index=mice_ids)
similarity_matrix = df_similarity.values
clustergrid = sns.clustermap(similarity_matrix, method='average', cmap='jet', vmax=1, vmin=-1, linewidths=0)
reordered_indices = clustergrid.dendrogram_row.reordered_ind
similarity_matrix_reordered = similarity_matrix[reordered_indices, :][:, reordered_indices]
mouse_list = df_similarity.index.tolist()
df_similarity = pd.DataFrame(similarity_matrix_reordered,
                         columns=np.array(mouse_list)[reordered_indices],
                         index=np.array(mouse_list)[reordered_indices])
sort_col = np.array(mouse_list)[reordered_indices]
heatmap = create_heatmap(df_similarity)
print("saving ./images/Fig_S3-A.png")
heatmap.save("images/Fig_S3-A.png")



plotting.plot_roi(DSURQE_thalamus_mask,bg_img=DSURQE_img)
plotting.plot_roi(DSURQE_mask,bg_img=DSURQE_img)

tstat = masking.apply_mask(all_runs, DSURQE_thalamus_mask)

df_similarity = pd.DataFrame(np.corrcoef(tstat),columns=mice_ids,
                            index=mice_ids)
similarity_matrix = df_similarity.values
clustergrid = sns.clustermap(similarity_matrix, method='average', cmap='jet', vmax=1, vmin=-1, linewidths=0)
reordered_indices = clustergrid.dendrogram_row.reordered_ind
similarity_matrix_reordered = similarity_matrix[reordered_indices, :][:, reordered_indices]
mouse_list = df_similarity.index.tolist()
df_similarity = pd.DataFrame(similarity_matrix_reordered,
                         columns=np.array(mouse_list)[reordered_indices],
                         index=np.array(mouse_list)[reordered_indices])
sort_col = np.array(mouse_list)[reordered_indices]
heatmap = create_heatmap(df_similarity)
print("saving ./images/Fig_S3-B.png")
heatmap.save("images/Fig_S3-B.png")