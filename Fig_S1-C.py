from scipy import ndimage
from nilearn import image
import altair as alt
import pandas as pd
import numpy as np
import os
from nilearn import image

PL_mice_mask_CCF_path = ...  #path to injection mask in CCFv3 space for PL group
Sub_mice_mask_CCF_path = ... #path to injection mask in CCFv3 space for Sub group



def create_scatter_plot(x_axis, y_axis,circle_radius=0.75,color='steelblue'):

    chart_width = 100
    chart_height = 100

    x_axis_value = axis_data[x_axis]
    y_axis_value = axis_data[y_axis]
    
    x_domain=[x_axis_value[0],x_axis_value[-1]]
    y_domain=[y_axis_value[0],y_axis_value[-1]]
    
    x_range = x_domain[-1] - x_domain[0]
    y_range = y_domain[-1] - y_domain[0]

    x_scale = alt.Scale(domain=x_domain,nice=False)
    y_scale = alt.Scale(domain=y_domain,nice=False)
    
    x_axis_ = alt.Axis(values=x_axis_value, tickMinStep=0.5, titleFontSize=8)
    y_axis_ = alt.Axis(values=y_axis_value, tickMinStep=0.5, titleFontSize=8)
    
    x_pixel_per_unit = chart_width / x_range
    y_pixel_per_unit = chart_height / y_range
    
    pixel_per_unit = min(x_pixel_per_unit, y_pixel_per_unit)
    
    def radius_to_area(radius, pixel_per_unit):
        radius_in_pixels = radius * pixel_per_unit
        area_in_pixels = np.pi * (radius_in_pixels ** 2)
        return area_in_pixels

    
    chart = alt.Chart(data).mark_circle(size=20, opacity=0.8).encode(
        x=alt.X(x_axis, title=x_axis,
                axis=x_axis_,
                scale=x_scale
               ),
        y=alt.Y(y_axis, title=y_axis, 
                axis=y_axis_,
                scale=y_scale
               ),
        tooltip=['Anterior-Posterior', 'Medial-Lateral', 'Dorsal-Ventral']
        ,color=alt.value(color) 
    ).properties(
        width=chart_width,
        height=chart_height
    )
    print("x: ",cross_dot_coor[x_axis]," y: ",cross_dot_coor[y_axis])

    circle = alt.Chart(pd.DataFrame({'x': [cross_dot_coor[x_axis]], 'y': [cross_dot_coor[y_axis]]})).mark_circle(
    size=radius_to_area(circle_radius,pixel_per_unit), opacity=0.3,strokeWidth=0.5, stroke='black').encode(
        x=alt.X('x:Q',axis=x_axis_, scale=x_scale),
        y=alt.Y('y:Q',axis=y_axis_, scale=y_scale)
        ,color=alt.value(color) 
    )


    x_mark = alt.Chart(pd.DataFrame({'x': [cross_dot_coor[x_axis]], 'y': [cross_dot_coor[y_axis]]})).mark_text(text='X', fontSize=8, ).encode(
        x=alt.X('x:Q',axis=x_axis_, scale=x_scale),
        y=alt.Y('y:Q',axis=y_axis_, scale=y_scale)
    )

    return chart + circle + x_mark




coords_CCF = []


for inject_filename in os.listdir(PL_mice_mask_CCF_path): 
    injection_mask = image.load_img(os.path.join(PL_mice_mask_CCF_path, inject_filename))
    injection_mask_data = np.array(injection_mask.dataobj)
    center = np.array(ndimage.center_of_mass(injection_mask_data)).astype('int')
    coord_ = image.coord_transform(center[0], center[1], center[2], image.load_img(injection_mask).affine)
    coords_CCF.append(coord_)


data = pd.DataFrame({
    'Anterior-Posterior': np.array(coords_CCF)[:, 1],
    'Medial-Lateral': np.array(coords_CCF)[:, 0],
    # convert to bergma coordination bergma start at 4.77
    'Dorsal-Ventral': 4.77-np.array(coords_CCF)[:, 2],
})

def convert_to_integ(x):
    if x < 0:
        if x > -1:
            return np.floor(x * 10) / 10
    else:
        if x<1:
            return np.ceil(x * 10) / 10
        else:
           return np.round(x / 0.5) * 0.5 
    return int(x / 0.5) * 0.5

cross_dot_coor={}
for name,value in data.items():
    cross_dot_coor[name] = np.mean(value).round(2)

print("cross_dot_coor: ",cross_dot_coor)

axis_data = {
     'Anterior-Posterior': [0.5,1.0, 1.5,2.0, 2.5,3.0],
    'Medial-Lateral': [-1.0,-0.5,0,0.5, 1.0, 1.5],
    'Dorsal-Ventral': [1.5,2.0, 2.5,3.0, 3.5,4.0]
}


chart_ap_ml = create_scatter_plot('Anterior-Posterior', 'Medial-Lateral', color="#2BA02B")
chart_ap_dv = create_scatter_plot('Anterior-Posterior', 'Dorsal-Ventral',color="#2BA02B")
chart_ml_dv = create_scatter_plot('Medial-Lateral', 'Dorsal-Ventral',color="#2BA02B")

final_chart = alt.hconcat(chart_ap_ml, chart_ap_dv, chart_ml_dv).properties(
    title="Prl injections"
).configure_title(fontSize=15, font='Arial', anchor='middle', color='black').configure_axis(
    labelFontSize=6,  
    gridWidth=0.4,  
    domainWidth=0.3   # axis width
)

print("saving...images/Fig_S1-C_PL.html")
final_chart.save('images/Fig_S1-C_PL.html')




coords_CCF = []

for inject_filename in os.listdir(Sub_mice_mask_CCF_path): 
    injection_mask = image.load_img(os.path.join(Sub_mice_mask_CCF_path, inject_filename))
    injection_mask_data = np.array(injection_mask.dataobj)
    center = np.array(ndimage.center_of_mass(injection_mask_data)).astype('int')
    coord_ = image.coord_transform(center[0], center[1], center[2], image.load_img(injection_mask).affine)
    coords_CCF.append(coord_)



data = pd.DataFrame({
    'Anterior-Posterior': np.array(coords_CCF)[:, 1],
    'Medial-Lateral': np.array(coords_CCF)[:, 0],
    'Dorsal-Ventral': 4.77-np.array(coords_CCF)[:, 2],
})
cross_dot_coor={}
for name,value in data.items():
    cross_dot_coor[name] = np.mean(value).round(2)

print("cross_dot_coor: ",cross_dot_coor)

axis_data = {
    'Anterior-Posterior': [-4.0,-3.5,-3.0,-2.5,-2.0,-1.5],
    'Medial-Lateral': [0.0,0.5, 1.0, 1.5,2.0,2.5],
    'Dorsal-Ventral': [0.5, 1.0, 1.5,2.0,2.5,3.0]
}

chart_ap_ml = create_scatter_plot('Anterior-Posterior', 'Medial-Lateral',circle_radius=0.75)
chart_ap_dv = create_scatter_plot('Anterior-Posterior', 'Dorsal-Ventral',circle_radius=0.75)
chart_ml_dv = create_scatter_plot('Medial-Lateral', 'Dorsal-Ventral',circle_radius=0.75)

final_chart = alt.hconcat(chart_ap_ml, chart_ap_dv, chart_ml_dv).properties(
    title="SUB injections"
).configure_title(fontSize=15, font='Arial', anchor='middle', color='black').configure_axis(
    labelFontSize=6,  
    gridWidth=0.4,  
    domainWidth=0.3   
)

print("saving...images/Fig_S1-D_Sub.html")
final_chart.save('images/Fig_S1-D_Sub.html')