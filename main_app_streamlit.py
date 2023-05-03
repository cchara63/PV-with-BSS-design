# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.
# 
import base64
from dash import Dash, html, dcc
import plotly.express as px

from plotly.subplots import make_subplots
import plotly.graph_objects as go

from dash.dependencies import Input, Output, State

import pandas as pd # use for xlsx insertion

import numpy as np
import math

import streamlit as st

st.set_page_config(page_title="EPyT viewer using streamlit",
                   layout="wide")
st.sidebar.title("EPyT - Viewer")
st.sidebar.info(
    """
    The EPANET-Python Toolkit is an open-source software, originally developed by the KIOS Research and 
    Innovation Center of Excellence, University of Cyprus which operates within the Python environment,
    for providing a programming interface for the latest version of EPANET, a hydraulic and quality modeling 
    software created by the US EPA, with Python, a high-level technical computing software. The goal of the 
    EPANET Python Toolkit is to serve as a common programming framework for research and development in the 
    growing field of smart water networks.
    
    EPyT GitHub:  <https://github.com/KIOS-Research/EPyT>
    Web App repository: <https://github.com/Mariosmsk/streamlit-epyt-viewer>
    """
)

df = pd.read_excel('pv_load_profile.xlsx')

# images
image_kios = 'kioslogo.png'
encoded_image_kios = base64.b64encode(open(image_kios, 'rb').read()).decode()
image_WiseWire = 'WiseWire.png'
encoded_image_WiseWire = base64.b64encode(open(image_WiseWire, 'rb').read()).decode()
image_anthropouthkia = 'anthropouthkia.png'
encoded_image_anthropouthkia = base64.b64encode(open(image_anthropouthkia, 'rb').read()).decode()
image_pv = 'pv.png'
encoded_image_pv = base64.b64encode(open(image_pv, 'rb').read()).decode()
image_bat = 'bat.png'
encoded_image_bat = base64.b64encode(open(image_bat, 'rb').read()).decode()

app = Dash(__name__)

fig = make_subplots(rows=4, cols=1)

options = [
    {'label': 'Winter', 'value': 'opt1'},
    {'label': 'Spring', 'value': 'opt2'},
    {'label': 'Summer', 'value': 'opt3'},
    {'label': 'Autumn', 'value': 'opt4'}
]# option for drobdown menu

plan_options = [
    {'label': 'Net-Metering', 'value': 'opt1'},
    {'label': 'Net-Billing', 'value': 'opt2'},
    
]# option for drobdown menu

app.layout = html.Div(children=[
    # images
    html.Img(src='data:image/jpg;base64,{}'.format(encoded_image_kios),
        style={'height': 80, 'width': 240,},),
    html.Img(src='data:image/jpg;base64,{}'.format(encoded_image_WiseWire),
        style={'height': 80, 'width': 200,"float": "right",'position': 'absolute','zIndex': '1','right': 10},), 
    html.Img(src='data:image/jpg;base64,{}'.format(encoded_image_anthropouthkia),
        style={'height': 80, 'width': 100,"float": "right",
        'position': 'absolute', 'top': 200, 'left': '300px'},), 
    html.Img(src='data:image/jpg;base64,{}'.format(encoded_image_pv),
        style={'height': 100, 'width': 150,"float": "right",
        'position': 'absolute', 'top':'23.5%', 'left': 280,},),
    html.Img(src='data:image/jpg;base64,{}'.format(encoded_image_bat),
        style={'height': 80, 'width': 200,"float": "right",
        'position': 'absolute', 'top': 700, 'left': 250},),
    # titles-lables 
    html.H1(children='''People are living at home'''),
    html.H1(children='''Photovoltaic installed value (kW)''',
        style={'position': 'absolute', 'top': 350, 'left': '10px'}),
    html.H1(children='''Plan:''',
        style={'position': 'absolute', 'top': '20.5%', 'left': '12%',"fontSize":"26px"}),
    html.H1(children='''Battery Storage System (kW/kWh)''',
        style={'position': 'absolute', 'top': '800px', 'left': '300px',
        'position': 'absolute', 'top': 600, 'left': 10}),
    html.H1(children='''Lower Your Electricity Bill!''',
        style={'position': 'absolute', 'top': '-1%', 'left': '23%',"fontSize":"46px"}),
    html.H1(children='''Season:''',
        style={'position': 'absolute', 'top': '0.5%', 'left': 1070,'position': 'absolute','zIndex': '1'}),        
    html.H1(children='''Energy cosumption cost''',
        style={'position': 'absolute', 'top': 100, 'left': 600,"fontSize":"26px"}),   
    html.H1(children='''Cost with PV installation''',
        style={ 'position': 'absolute', 'top': 300, 'left': 600,"fontSize":"26px"}), 
    html.H1(children='''Cost with PV & BSS installation''',
        style={ 'position': 'absolute', 'top': 500, 'left': 600,"fontSize":"26px"}),     
    html.H1(children='''Investement cost & payback time''',
        style={ 'position': 'absolute', 'top': 700, 'left': 600,"fontSize":"26px"}),          
    # plots
    dcc.Graph(id='graph_id',
        style={'height': 950,'width': 1000,"display": "block","float": "right",'position': 'relative', 'top': -180, 'left': 120,'zIndex': '0',
         },
        figure=fig , 
        
        ),       
    # inputs   
    dcc.Input(id='people_n', type='number', value=0, style={'height': 50, 'width': 100,"float": "right",
        'position': 'absolute', 'top': 200, 'left': 10, "fontSize": "28px"}),
    dcc.Input(id='pv_kw', type='number', value=0,style={'height': 50, 'width': 100,"float": "right",
        'position': 'absolute', 'top': 430, 'left': 10, "fontSize": "28px"}),
    dcc.Input(id='battery_kw_kwh', type='number', value=0,style={'height': 50, 'width': 100,"float": "right",
        'position': 'absolute', 'top': 670, 'left': 10, "fontSize": "28px"}),
    # buttons
    #html.Button('Reload', id='reload-button1',n_clicks=0,style={'position': 'absolute', 'top': 200, 'left': 150}), 
    #html.Button('Submit', id='submit-button',style={'position': 'absolute', 'top': 430, 'left': 150}), 
    #html.Button('Reload', id='reload-button3',style={'position': 'absolute', 'top': 670, 'left': 150}), 
    # output print 
    html.Div(id='load_cost', children='',style={'position': 'absolute', 'top': 170, 'left':'33%',"fontSize":"22px",'backgroundColor':'#ffffff' }),
    html.Div(id='price_with_pv', children='',style={'position': 'absolute', 'top': 370, 'left':'33%',"fontSize":"22px",'backgroundColor':'#ffffff'}),
    html.Div(id='price_with_pv_bat', children='',style={'position': 'absolute', 'top': 570, 'left':'33%',"fontSize":"22px",'backgroundColor':'#ffffff'}),
    html.Div(id='return_years', children='',style={'position': 'absolute', 'top': 800, 'left':'33%',"fontSize":"22px",'backgroundColor':'#ffffff'}),
    html.Div(id='total_cost', children='',style={'position': 'absolute', 'top': 770, 'left':'33%',"fontSize":"22px",'backgroundColor':'#ffffff'}),
    html.Div(id='import_price', children='',style={'position': 'absolute', 'top':'15%', 'left':'10%',"fontSize":"22px",'backgroundColor':'#ffffff'}),
    html.Div(id='export_price', children='',style={'position': 'absolute', 'top': '29%', 'left':'10%',"fontSize":"22px",'backgroundColor':'#ffffff'}), #fef0d9
    
    # drobdown 'options-align': 'center',
    dcc.Dropdown( id='my_dropdown', options=options, value='opt1',
        style={ 'position': 'absolute', 'width': 120, 'top':'1.7%', 'left':'60%',"fontSize":"16px"}), 
    # DROPDOWN for plan
    dcc.Dropdown( id='plan_dropdown', options=plan_options, value='opt1', 
        style={  'position': 'relative' , 'width': 150, 'top':'13%', 'left':'15%',"fontSize":"16px"},
        #dropdownStyle={ 'position': 'absolute','optionsAlign': 'center', 'width': 140, 'top':210, 'left':'15%'},
        ), 
       
], style={'position': 'absolute', 'top': 0, "background":'#9a9a9a', 'height': 2000, 'width': 1880,'zIndex': '-1',}
)
 
@app.callback(
    Output('graph_id', 'figure'),
    Output('load_cost', 'children'),
    Output('price_with_pv', 'children'),
    Output('price_with_pv_bat', 'children'),
    Output('return_years', 'children'),
    Output('total_cost', 'children'),   
    Output('import_price', 'children'),
    Output('export_price', 'children'), 
    #Output('output-season', 'children'),
    #Input('reload-button1', 'n_clicks'),
    Input('people_n', 'value'),
    Input('pv_kw','value'),
    Input('battery_kw_kwh','value'),
    [Input('my_dropdown', 'value')], 
    [Input('plan_dropdown', 'value')],
    )
def update_figure(people_n,pv_kw,battery_kw_kwh,my_dropdown,plan_dropdown): #n_clicks,
       
    fig = make_subplots(rows=4, cols=1,
     subplot_titles=("Load Power (W)", "Photovoltaic Power (W)", "Battery Power (W)", "Grid:Total Power (W)"))
    
    

    # specify season for plots
    if my_dropdown == 'opt1':
        load_season = 'load_winter'
        pv_season = 'pv_winter'
    elif my_dropdown == 'opt2':
        load_season = 'load_spring'
        pv_season = 'pv_spring'
    elif my_dropdown == 'opt3':
        load_season = 'load_summer'
        pv_season = 'pv_summer'
    else:# my_dropdown == 'opt4':
        pv_season = 'pv_autumn'
        load_season = 'load_autumn'
        
    # specify PV PLANfor plots
    kwh_c_in = 0.36 # Euro per kWh
    if plan_dropdown == 'opt1':
        pv_plan = 'Net-Metering'
        kwh_c_out = 0.32 # Euro per kWh
    else: #plan_dropdown == 'opt6':
        pv_plan = 'Net-Billing'
        kwh_c_out = 0.18 # Euro per kWh
        
    # find the multply factor    
    if people_n == 0:
        peop_mul =0
    else: 
        peop_mul=((people_n-1)*0.538 + 1)     
    
    # 1st figure - load
    fig.append_trace(go.Scatter(
        x=df['time'],
        y= peop_mul * df[load_season],),
        row=1, col=1#plot_bgcolor="#e6f3ff", paper_bgcolor="#e6f3ff"
        )
    # Set title for subplot (3, 1)
    #fig.update_layout( {'title':"My Plot Title"}, row=2,col=1 )    
        
        
        
    # 2nd figure - PV  #print('pv Data')   #print(df[pv_season].dtype)
    fig.append_trace(go.Scatter(
        x=df['time'],
        y=pv_kw*df[pv_season],), 
        row=2, col=1)
    
    # initial cost, total sum of load cost without any PV or Battery 
    total_load = 0
    load = df['load_winter']*90 + df['load_spring']*92 + df['load_summer']*92 + df['load_autumn']*91
    for element in load:
        total_load +=  element
    
    total_cost = total_load/1000/4* kwh_c_in * peop_mul
    
            #if (app.Switch_PV.Value == "On") % if swtich it is off
    # Grid power: load + PV
    Pgrid_wi = df['load_winter']*peop_mul - df['pv_winter']*pv_kw
    Pgrid_sp = df['load_spring']*peop_mul - df['pv_spring']*pv_kw
    Pgrid_su = df['load_summer']*peop_mul - df['pv_summer']*pv_kw
    Pgrid_au = df['load_autumn']*peop_mul - df['pv_autumn']*pv_kw
            # else
                # Pgrid1 = home_his_Win ;
                # Pgrid2 = home_his_Spr ;
                # Pgrid3 = home_his_Sum ;
                # Pgrid4 = home_his_Aut ;               
            # end   
            
    ## cost with PV
    cost_with_PV = 0
    for i in range(0,96,1):
        if Pgrid_wi[i]>0: 
            epkw_wi = kwh_c_in # Euro per kWh, consumption charge
        else:
            epkw_wi = kwh_c_out  # import 36 cent, export 32 or 18 cent
            
        if Pgrid_sp[i]>0: 
            epkw_sp = kwh_c_in
        else:
            epkw_sp = kwh_c_out  # import ..
            
        if Pgrid_su[i]>0: 
            epkw_su = kwh_c_in
        else:
            epkw_su = kwh_c_out #import ..
            
        if Pgrid_au[i]>0: 
            epkw_au = kwh_c_in
        else:
           epkw_au = kwh_c_out #import ..
        
        #cost of energy consumption with PV
        cost_with_PV = cost_with_PV + Pgrid_wi[i]/1000/4*epkw_wi*90 + Pgrid_sp[i]/1000/4*epkw_sp*92 + Pgrid_su[i]/1000/4*epkw_su*92 + Pgrid_au[i]/1000/4*epkw_au*91
    
    # BSS implimantation
    Pbss_wi = np.zeros((96,1))     
    Pbss_sp = np.zeros((96,1)) 
    Pbss_su = np.zeros((96,1))
    Pbss_au = np.zeros((96,1))
    SOC_BSS_wi = np.zeros((97,1))
    SOC_BSS_sp = np.zeros((97,1))
    SOC_BSS_su = np.zeros((97,1))
    SOC_BSS_au = np.zeros((97,1))
       
            # if (app.Switch_BSS.Value == "On") % if swtich it is off
    SOC_BSS_wi[0,0] = 50
    SOC_BSS_sp[0,0] = 50
    SOC_BSS_su[0,0] = 50
    SOC_BSS_au[0,0] = 50
    inv_w = battery_kw_kwh * 1000; # W inverter nominal value 
    cap_wh = battery_kw_kwh*1000;  # kWh 
    delta_t = 0.25; # digmatolipsia 15 min                
    
    # loop for SOC calculation
    i=0    
    while i<95:
        # set the Pbss to push the grid power to zero
        
        # from (96,) to (96,1) matrix        
        Pgrid_wi = np.array(Pgrid_wi).reshape((96, 1)) 
        Pgrid_sp = np.array(Pgrid_sp).reshape((96, 1)) 
        Pgrid_su = np.array(Pgrid_su).reshape((96, 1)) 
        Pgrid_au = np.array(Pgrid_au).reshape((96, 1)) 
        
        # initial value of BSS, before inverter and energy capacity limits 
        Pbss_wi[i,0] = Pgrid_wi[i,0]     
        Pbss_sp[i,0] = Pgrid_sp[i,0]
        Pbss_su[i,0] = Pgrid_su[i,0] 
        Pbss_au[i,0] = Pgrid_au[i,0]
                
        # Inverter limits control
        if (abs(Pbss_wi[i,0]) > inv_w): 
            Pbss_wi[i,0] = math.copysign(1,Pbss_wi[i,0]) * inv_w
        if (abs(Pbss_sp[i,0]) > inv_w): 
            Pbss_sp[i,0] = math.copysign(1,Pbss_sp[i,0]) * inv_w
        if (abs(Pbss_su[i,0]) > inv_w): 
            Pbss_su[i,0] = math.copysign(1,Pbss_su[i,0]) * inv_w
        if (abs(Pbss_au[i,0]) > inv_w): 
            Pbss_au[i,0] = math.copysign(1,Pbss_au[i,0]) * inv_w 
    
        # BSS limits control
        if ( (Pbss_wi[i,0] < 0) & (SOC_BSS_wi[i,0] >= 100)):
            Pbss_wi[i,0] = 0
        elif( (Pbss_wi[i,0] > 0) & (SOC_BSS_wi[i,0] <= 20) ):
            Pbss_wi[i,0] = 0
        if ( (Pbss_sp[i,0] < 0) & (SOC_BSS_sp[i,0] >= 100) ):
            Pbss_sp[i,0] = 0
        elif( (Pbss_sp[i,0] > 0) & (SOC_BSS_sp[i,0] <= 20) ):
            Pbss_sp[i,0] = 0
        if  ( (Pbss_su[i,0] < 0) & (SOC_BSS_su[i,0] >= 100)):
            Pbss_su[i,0] = 0
        elif( (Pbss_su[i,0] > 0) & (SOC_BSS_su[i,0] <= 20) ):
            Pbss_su[i,0] = 0
        if  ( (Pbss_wi[i,0] < 0) & (SOC_BSS_wi[i,0] >= 100)):
            Pbss_wi[i,0] = 0
        elif( (Pbss_au[i,0] > 0) & (SOC_BSS_au[i,0] <= 20) ):
            Pbss_au[i,0] = 0
                                        
        # BSS behavior - lipoun oi apolies
        if cap_wh != 0:       
            SOC_BSS_wi[i+1,0] = SOC_BSS_wi[i,0] - Pbss_wi[i,0]*delta_t / cap_wh*100
            SOC_BSS_sp[i+1,0] = SOC_BSS_sp[i,0] - Pbss_sp[i,0]*delta_t / cap_wh*100
            SOC_BSS_su[i+1,0] = SOC_BSS_su[i,0] - Pbss_su[i,0]*delta_t / cap_wh*100
            SOC_BSS_au[i+1,0] = SOC_BSS_au[i,0] - Pbss_au[i,0]*delta_t / cap_wh*100
        i = i+1
        #while end

             #if (app.Switch_BSS.Value == "On") % if swtich it is off 
             
    # Ιnsert BSS energy in total grid Power             
    Pgrid_wi = Pgrid_wi - Pbss_wi
    Pgrid_sp = Pgrid_sp - Pbss_sp
    Pgrid_su = Pgrid_su - Pbss_su
    Pgrid_au = Pgrid_au - Pbss_au
       
    if my_dropdown == 'opt1': #if (app.SeasonDropDown.Value == "1")
        fig3 = Pbss_wi
        fig4 = Pgrid_wi
               
    if my_dropdown == 'opt2':    # elseif (app.SeasonDropDown.Value == "2")
        fig3 = Pbss_sp
        fig4 = Pgrid_sp
               
    if my_dropdown == 'opt3':     # elseif(app.SeasonDropDown.Value =="3")
       fig3 = Pbss_su
       fig4 =Pgrid_su
               
    if my_dropdown == 'opt4':     # else
        fig3 = Pbss_au
        fig4 =Pgrid_au  
    
    # figure 3 #print('fig3') #print(fig3.dtype) #print(fig3.shape)#print(df['time'].shape)
    fig3 = fig3.flatten()    
    fig.append_trace(
        go.Scatter(x=df['time'],
                   y=fig3), # [10, 11*people_n, 12]
                    
        row=3, col=1)
    ## end of BSS figure
    
    # figure 4
    fig4 = fig4.flatten()
    fig.append_trace(go.Scatter(
        x=df['time'],
        y=fig4
    ), row=4, col=1)
    
    fig.update_layout(
        plot_bgcolor="#282828",
        paper_bgcolor="#9a9a9a",
        xaxis = dict(gridcolor="#4b4b4b"),
        yaxis = dict(gridcolor="#4b4b4b"),
        xaxis2 = dict(gridcolor="#4b4b4b"),
        yaxis2 = dict(gridcolor="#4b4b4b"),
        xaxis3 = dict(gridcolor="#4b4b4b"),
        yaxis3 = dict(gridcolor="#4b4b4b"),
        xaxis4 = dict(gridcolor="#4b4b4b"),
        yaxis4 = dict(gridcolor="#4b4b4b"),
        font=dict( #family = :"Arial",
                   #size=18,
                   color="white")
    ) 
     
    # # grid load with PV installation
    # p_grid_pv_wi = df['load_winter']*peop_mul-df['pv_winter']*pv_kw 
    # p_grid_pv_sp = df['load_spring']*peop_mul-df['pv_spring']*pv_kw 
    # p_grid_pv_su = df['load_summer']*peop_mul-df['pv_summer']*pv_kw 
    # p_grid_pv_au = df['load_autumn']*peop_mul-df['pv_autumn']*pv_kw
    
    #for row in 
    
    # Update the x-axis layout
    fig.update_xaxes(tickmode='linear', dtick=4)
    
    
      
    ## cost with PV and Battery
    cost_with_PV_Bat = 0
    for i in range(0,96,1):
        if Pgrid_wi[i]>0: 
            epkw_wi = kwh_c_in 
        else:
            epkw_wi = kwh_c_out  # import 28 cent, export 16 cent
            
        if Pgrid_sp[i]>0: 
            epkw_sp = kwh_c_in
        else:
            epkw_sp = kwh_c_out  # import 28 cent, export 16 cent
        if Pgrid_su[i]>0: 
            epkw_su = kwh_c_in
        else:
            epkw_su = kwh_c_out #import 28 cent, export 16 cent
        if Pgrid_au[i]>0: 
            epkw_au = kwh_c_in
        else:
           epkw_au = kwh_c_out #import 28 cent, export 16 cent
        
        #cost PV and Battery
        cost_with_PV_Bat = cost_with_PV_Bat + Pgrid_wi[i]/1000/4*epkw_wi*90 + Pgrid_sp[i]/1000/4*epkw_sp*92 + Pgrid_su[i]/1000/4*epkw_su*92 + Pgrid_au[i]/1000/4*epkw_au*91  

        # c_f for basic cost of a PV SYSTEM
        if pv_kw == 0:
            c_f = 0
        else: c_f =1
         
        Total_invest_cost = c_f*(1000+pv_kw*1100) + battery_kw_kwh*1100; # investement cost
        
        if (total_cost-cost_with_PV_Bat) != 0:
            Return_Time = float(Total_invest_cost/(total_cost - cost_with_PV_Bat))
        else:
            Return_Time = float(0)
            
    
    # check if the reload button was clicked
    #if n_clicks is not None:
        # reset the interval timer to trigger a page reload
    return (fig,
            f'The annual cost of load is {int(total_cost)} €', 
            f'The annual cost with PV is {int(cost_with_PV)} €', 
            f'The annual cost with PV and Battery is {int(cost_with_PV_Bat)} €', 
            f'Investement payback time { "{:.1f}".format(Return_Time) } years',
            f'Total cost for energy upgrade is {int(Total_invest_cost)} €',
            f'Charge for import energy {  "{:.2f}".format(kwh_c_in)  } €/kWh',
            f'Charge for export energy {"{:.2f}".format(kwh_c_out)} €/kWh',            
            #f'You have selected {my_dropdown}'
    )
    #else:
    #    return {'data': [], 'layout': {}}
          
if __name__ == '__main__':
    app.run_server(debug=True)
    
    