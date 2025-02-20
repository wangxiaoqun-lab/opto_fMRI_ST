import pandas as pd
import altair as alt

Sub_test_r_csv =... # the correlation result of FD vs DVAS for sub test group
Sub_con_r_csv  =... # the correlation result of FD vs DVAS for sub con group
PL_test_r_csv  =... # the correlation result of FD vs DVAS for PL test group
PL_con_r_csv   =... # the correlation result of FD vs DVAS for PL con group

def voilint_plot_pca_selection(df,color_scale,num_format='.2f'):
    

    # Define global min and max
    global_min = 0
    global_max = 1
    
    wdith_size=30
    height_size=100
    
    # Define the density plot
    density_plot = alt.Chart().transform_density(
        'r',
        groupby=['denoise-method'],
        as_=['r', 'density']
    ).mark_area(orient='horizontal',size=10).encode(
        x=alt.X('density:Q', stack='center', title=None, axis=alt.Axis(labels=False, values=[0], grid=False, ticks=True)),
        y=alt.Y('r:Q'),color=alt.Color('denoise-method:N', scale=color_scale)
    ).properties(
        width=wdith_size,
        height=height_size
    )
    
    density_plot = alt.Chart(df).transform_density(
        'r',
        groupby=['denoise-method'],
        as_=['r', 'density'],
    ).mark_area(orient='horizontal', fill='white', strokeWidth=1.5).encode(
        x=alt.X('density:Q', stack='center', title=None, axis=alt.Axis(labels=False, values=[0], grid=False, ticks=True)),
        y=alt.Y('r:Q', scale=alt.Scale(domain=[global_min, global_max])),
        stroke=alt.Color('denoise-method:N', scale=color_scale)
    ).properties(
        width=wdith_size,
        height=height_size
    )
    
    # Define the boxplot
    boxplot = alt.Chart().mark_boxplot(color="black",size=6).encode(
        x=alt.X('denoise-method:O', sort=sort_col, axis=alt.Axis(orient='bottom', title=None, ticks=False)),
        y='r:Q',
    ).properties(
        width=wdith_size,
        height=height_size
    )
    median_text = alt.Chart().transform_aggregate(
        median_r='median(r)',
        groupby=['denoise-method']
    ).mark_text(
        align='center',
        baseline='bottom',
        dx=2,
        dy=0
    ).encode(
        y=alt.value(0),
        x=alt.X('denoise-method:O', stack='center', title=None, axis=alt.Axis(labels=False,grid=False, ticks=False)),
        text=alt.Text('median_r:Q', format=num_format),
        color=alt.Color('denoise-method:N', scale=color_scale)
    ).properties(
        width=wdith_size,
        height=height_size
    )
    
    
    # Layer the plots and facet without column titles
    layered_plot = alt.layer(density_plot,boxplot,median_text, data=df).facet(
        column=alt.Column('denoise-method:N', sort=sort_col, header=alt.Header(title=None, labels=False))
    ).resolve_scale(
        x=alt.ResolveMode('independent')
    ).configure_facet(
        spacing=2
    ).configure_view(
        stroke=None
    )
    
    return layered_plot

sort_col = ['raw', '6-clean', '12-clean', '10-pc', '20-pc', '30-pc', '40-pc', '50-pc', '60-pc', '70-pc', '80-pc', '90-pc', '100-pc']
con_color_scale = alt.Scale(
    domain=sort_col,
    range=[
        'black', 'magenta', 'magenta', 
        '#f52121', '#f74e4e', '#F75454', '#F75353', 
        '#f97b7b', '#F97F7F', '#fba8a8', '#fdd5d5', 
        '#ffe6e6', '#FFF0EF'
    ]
)
df = pd.read_csv(Sub_con_r_csv)
df['denoise-method'] = [i.split("_")[0] for i in df['denoise-method']]
sub_con_chart = voilint_plot_pca_selection(df,con_color_scale)
print("Saving.. images/Fig_S2-D-sub_con_chart.html")
sub_con_chart.save("images/Fig_S2-D-sub_con_chart.html")


df = pd.read_csv(PL_con_r_csv)
df['denoise-method'] = [i.split("_")[0] for i in df['denoise-method']]
pl_con_chart = voilint_plot_pca_selection(df,con_color_scale)
print("Saving.. images/Fig_S2-D-pl_con_chart.html")
pl_con_chart.save("images/Fig_S2-D-pl_con_chart.html")

sub_test_color_scale = alt.Scale(
    domain=[
        'raw', '6-clean', '12-clean', 
        '10-pc', '20-pc', '30-pc', '40-pc', '50-pc', '60-pc', '70-pc', '80-pc', '90-pc', '100-pc'
    ],
    range=[
        'black',  # raw
        'magenta', 'magenta',  # 6-clean, 12-clean
        '#005b96', '#03396c', '#011f4b',  # 10-pc to 40-pc
        '#03396c', '#005b96','#4C84B1', '#5B8EB1','#6497b1', '#b3cde3', '#e6f2ff'  # 50-pc to 100-pc
    ]
)

df = pd.read_csv(Sub_test_r_csv)
df['denoise-method'] = [i.split("_")[0] for i in df['denoise-method']]
sub_test_chart = voilint_plot_pca_selection(df,sub_test_color_scale,'.3f')
print("Saving.. images/Fig_S2-D-sub_test_chart.html")
sub_test_chart.save("images/Fig_S2-D-sub_test_chart.html")

PL_test_color_scale = alt.Scale(
    domain=[
        'raw', '6-clean', '12-clean', 
        '10-pc', '20-pc', '30-pc', '40-pc', '50-pc', '60-pc', '70-pc', '80-pc', '90-pc', '100-pc'
    ],
    range=[
        'black',  # raw_r
        'magenta', 'magenta',  # 6-clean_r, 12-clean_r
        '#d5eebb', '#accb88', '#83a957', '#5b8725', '#387012', '#5b8725', '#83a957', '#accb88', '#d5eebb',"#E6FBD6" # 10-pc_r to 100-pc_r
    ]
)

df = pd.read_csv(PL_test_r_csv)
df['denoise-method'] = [i.split("_")[0] for i in df['denoise-method']]
pl_test_chart = voilint_plot_pca_selection(df,PL_test_color_scale)
print("Saving.. images/Fig_S2-D-pl_test_chart.html")
pl_test_chart.save("images/Fig_S2-D-pl_test_chart.html")
