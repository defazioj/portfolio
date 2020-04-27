# basic packages
import pandas as pd
import numpy as np
import warnings
from calendar import day_abbr


from plotly import graph_objs as go
from scipy.stats import skew
import plotly.offline as py
import matplotlib.pyplot as plt
import seaborn as sns

from fbprophet import Prophet
from fbprophet.diagnostics import cross_validation
from fbprophet.diagnostics import performance_metrics

# %matplotlib inline

# chart functions

def demo_charts(df):
    mask = (df['age'] <= 80) & (df['gender'] != 'unknown')

    # historgrams
    m = df[mask & (df['gender'] == 'male')]['age']
    f = df[mask & (df['gender'] == 'female')]['age']

    # pie chart
    dat = df[mask].gender.value_counts(normalize = True)

    # figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (20, 5))

    ax1.hist(m, bins = 63, alpha = 0.7, color = 'skyblue', label = 'Male', density = True)
    ax1.hist(f, bins = 63, alpha = 0.3, color = 'red', label = 'Female', density = True)
    ax1.text(50, 0.037, 'Median Age, Female: {:.0f}'.format(f.median()), fontsize = 14)
    ax1.text(50, 0.033, 'Median Age, Male: {:.0f}'.format(m.median()), fontsize = 14)    
    ax1.set_xlabel('Age', fontsize = 13)
    ax1.legend(loc = 'upper right', fontsize = 14)

    ax2.pie(dat.values, 
            labels = ['Male', 'Female'], 
            startangle = 180,
            colors = ['skyblue', 'pink'], 
            autopct='%1.0f%%', 
            textprops = {'fontsize' : 14})

    fig.tight_layout(pad=0.0);           

    
def demo_comm(df):

    mask = (df.index.hour.isin([7, 8, 9, 17, 18, 19])) & (df.index.weekday < 6) & (df['gender'] != 'unknown') & (df['age'] <= 80)

    # historgrams
    m = df[mask & (df['gender'] == 'male')]['age']
    f = df[mask & (df['gender'] == 'female')]['age']

    # pie chart
    dat = df[mask].gender.value_counts(normalize = True)

    # figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (20, 5))

    ax1.hist(m, bins = 63, alpha = 0.7, color = 'skyblue', label = 'Male', density = True)
    ax1.hist(f, bins = 63, alpha = 0.3, color = 'red', label = 'Female', density = True)
    ax1.text(50, 0.037, 'Median Age, Female: {:.0f}'.format(f.median()), fontsize = 14)
    ax1.text(50, 0.033, 'Median Age, Male: {:.0f}'.format(m.median()), fontsize = 14)    
    ax1.set_xlabel('Age', fontsize = 13)
    ax1.legend(loc = 'upper right', fontsize = 14)

    ax2.pie(dat.values, 
            labels = ['Male', 'Female'], 
            startangle = 180,
            colors = ['skyblue', 'pink'], 
            autopct='%1.0f%%', 
            textprops = {'fontsize' : 14})

    fig.tight_layout(pad=0.0);  
    
def demo_weeknd(df):
    
    mask = (df['age'] <= 80) & (df['gender'] != 'unknown')
    mask = (mask & (df.index.weekday > 5))

    # historgrams
    m = df[mask & (df['gender'] == 'male')]['age']
    f = df[mask & (df['gender'] == 'female')]['age']

    # pie chart
    dat = df[mask].gender.value_counts(normalize = True)

    # figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (20, 5))

    ax1.hist(m, bins = 63, alpha = 0.7, color = 'skyblue', label = 'Male', density = True)
    ax1.hist(f, bins = 63, alpha = 0.3, color = 'red', label = 'Female', density = True)
    ax1.text(50, 0.037, 'Median Age, Female: {:.0f}'.format(f.median()), fontsize = 14)
    ax1.text(50, 0.033, 'Median Age, Male: {:.0f}'.format(m.median()), fontsize = 14)    
    ax1.set_xlabel('Age', fontsize = 13)
    ax1.legend(loc = 'upper right', fontsize = 14)

    ax2.pie(dat.values, 
            labels = ['Male', 'Female'], 
            startangle = 180,
            colors = ['skyblue', 'pink'], 
            autopct='%1.0f%%', 
            textprops = {'fontsize' : 14})

    fig.tight_layout(pad=0.0);     
    
def night_riders(df):

    mask = (df.index.hour.isin([0, 1, 2, 3, 4, 21, 22, 23])) & (df['gender'] != 'unknown') & (df['age'] <= 80)

    # historgrams
    m = df[mask & (df['gender'] == 'male')]['age']
    f = df[mask & (df['gender'] == 'female')]['age']

    # pie chart
    dat = df[mask].gender.value_counts(normalize = True)

    # figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (20, 5))

    ax1.hist(m, bins = 63, alpha = 0.7, color = 'skyblue', label = 'Male', density = True)
    ax1.hist(f, bins = 63, alpha = 0.3, color = 'red', label = 'Female', density = True)
    ax1.text(50, 0.037, 'Median Age, Female: {:.0f}'.format(f.median()), fontsize = 14)
    ax1.text(50, 0.033, 'Median Age, Male: {:.0f}'.format(m.median()), fontsize = 14)    
    ax1.set_xlabel('Age', fontsize = 13)
    ax1.legend(loc = 'upper right', fontsize = 14)

    ax2.pie(dat.values, 
            labels = ['Male', 'Female'], 
            startangle = 180,
            colors = ['skyblue', 'pink'], 
            autopct='%1.0f%%', 
            textprops = {'fontsize' : 14})

    fig.tight_layout(pad=0.0);       
    
def demo_charts_r(df_r):    

    mask = (df_r.a_age <= 80) & (df_r.b_age <= 80)

    a = df_r[mask][['a_age', 'a_gender']].rename(columns = {'a_age' : 'age','a_gender' :'gender'})
    b = df_r[mask][['b_age', 'b_gender']].rename(columns = {'b_age' : 'age','b_gender' :'gender'})

    dat = a.append(b)

    # historgrams
    m = dat[(dat['gender'] == 'male')]['age']
    f = dat[(dat['gender'] == 'female')]['age']

    plt.rcParams.update({'font.size': 14})

    # figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (20, 5))

    ax1.hist(m, bins = 63, alpha = 0.7, color = 'skyblue', label = 'Male', density = True)
    ax1.hist(f, bins = 63, alpha = 0.3, color = 'red', label = 'Female', density = True)
    ax1.text(50, 0.047, 'Median Age, Female: {:.0f}'.format(f.median()), fontsize = 14)
    ax1.text(50, 0.04, 'Median Age, Male: {:.0f}'.format(m.median()), fontsize = 14)     
    ax1.set_xlabel('Age', fontsize = 15)
    ax1.legend(loc = 'upper right')

    mask = (dat['gender'] != 'unknown')
    ax2.pie(dat[mask].gender.value_counts(normalize = True),
            labels = ['Male', 'Female'],
            startangle = 180,
            colors = ['skyblue', 'pink'],
            autopct='%1.0f%%', 
            textprops = {'fontsize' : 14})
    fig.tight_layout(pad=0.0);  


def trend_charts(df):
    year_df = df[['tripduration']].groupby(df.index.year).count()
    day_df = df[['tripduration']].groupby(df.index.weekday).count()
    day_df['pct'] = day_df.tripduration / day_df.tripduration.sum()

    # restricing month_df to full years
    month_df = df['2014':'2019'][['tripduration']].groupby(df['2014':'2019'].index.month).count()
    month_df['pct'] = month_df.tripduration / month_df.tripduration.sum()
    day_df['day'] = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    month_df['month'] = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec']

    fig = plt.figure()
    plt.rcParams.update({'font.size': 18})
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (30, 7))

    # total data is downsampled by 10x
    ax1.bar(year_df.index, year_df.tripduration * 10 / 1000000)
    ax2.bar(month_df.month, month_df.pct)
    ax3.bar(day_df.day, day_df.pct)

    ax1.set_ylabel('Rides (Millions)')
    ax2.set_ylabel('Normalized')
    ax3.set_ylabel('Normalized');

    
def time_heatmap(df):

    df_heat = df[['tripduration']]
    df_heat['day_num'] = df.index.weekday
    df_heat['day_name'] = df.index.weekday_name
    df_heat['day_hour'] = df.index.hour

    fig, ax = plt.subplots(figsize = (30, 7))

    ax = sns.heatmap(df_heat.pivot_table(index = 'day_num', 
                                         columns = 'day_hour', 
                                         values = 'tripduration', 
                                         aggfunc = 'count'), 
                     cmap = 'Blues'
    )

    ax.set_xlabel('Hour of day', fontsize=22)
    ax.set_ylabel('Day of week', fontsize=22)
    ax.set_yticklabels(day_abbr[0:7])

    cbar = ax.collections[0].colorbar
    cbar.set_ticks([30000, 60000, 90000, 120000, 150000])

    # df is downsampled 10x
    cbar.set_ticklabels(['300k', '600k', '900k', '1.2M', '1.5M']);
    
def time_heatmap_r(df_r):   
    
    df_r['day_num'] = df_r.index.weekday
    df_r['day_name'] = df_r.index.weekday_name
    df_r['day_hour'] = df_r.index.hour

    f, ax = plt.subplots(figsize = (30, 7))

    ax = sns.heatmap(df_r.pivot_table(index = 'day_num', 
                                         columns = 'day_hour', 
                                         values = 'start_station_id', 
                                         aggfunc = 'count'), 
                     cmap = 'Blues'
    )

    ax.set_xlabel('Hour of day', fontsize=22)
    ax.set_ylabel('Day of week', fontsize=22)
    ax.set_yticklabels(day_abbr[0:7])

    cbar = ax.collections[0].colorbar
    cbar.set_ticks([4000, 8000, 12000, 16000, 20000, 24000])
    cbar.set_ticklabels(['4k', '8k', '12k', '16k', '20k', '24k']);  
        

def dist_oe(df_r, age_mean, age_var):
    
    mask = (df_r.a_age <= 80) & (df_r.b_age <= 80) & (df_r.a_gender != 'unknown') & (df_r.b_age != 'unknown')

    # expected and observed distributions
    e = np.random.normal(age_mean - age_mean, np.sqrt(age_var + age_var), df_r.shape[0]).round()
    o = df_r[mask].a_age - df_r[mask].b_age

    # figure
    fig, (ax1) = plt.subplots(1, 1, figsize = (10, 5))

    ax1.hist(e, bins = 80, alpha = 0.7, range = [-40, 40], color = 'skyblue', label = 'Expected', density = True)
    ax1.hist(o, bins = 80, alpha = 0.3, range = [-40, 40], color = 'red', label = 'Observed', density = True)
    ax1.text(-35, .125, 'Expected Dist SD: {:.1f}'.format(np.std(e)))
    ax1.text(-35, .11, 'Observed Dist SD: {:.1f}'.format(np.std(o)))
    ax1.set_xlabel('Age Difference', fontsize = 15)
    ax1.legend(loc = 'upper right');        
    

def dist_oes(df_r, df_s, age_mean, age_var):
    # adding our sampled dataset on our distributions visualization

    mask = (df_r.a_age <= 80) & (df_r.b_age <= 80) & (df_r.a_gender != 'unknown') & (df_r.b_age != 'unknown')

    # expected and observed distributions
    e = np.random.normal(age_mean - age_mean, np.sqrt(age_var + age_var), df_r.shape[0]).round()
    o = df_r[mask].a_age - df_r[mask].b_age

    # figure
    fig, (ax1) = plt.subplots(1, 1, figsize = (10, 5))

    ax1.hist(e, bins = 80, alpha = 0.7, range = [-40, 40], color = 'skyblue', label = 'Expected', density = True)
    ax1.hist(df_s['diff'], bins = 80, alpha = 0.3, range = [-40, 40], color = 'purple', label = 'Sampled', density = True)    
    ax1.hist(o, bins = 80, alpha = 0.3, range = [-40, 40], color = 'red', label = 'Observed', density = True)    
    ax1.text(-35, .125, 'Expected Dist SD: {:.1f}'.format(np.std(e)))
    ax1.text(-35, .11, 'Sampled Dist SD: {:.1f}'.format(np.std(df_s['diff'])))    
    ax1.text(-35, .095, 'Observed Dist SD: {:.1f}'.format(np.std(o)))
    ax1.set_xlabel('Age Difference', fontsize = 15)
    ax1.legend(loc = 'upper right'); 
    
    
def m_rmse(df, forecast):
    mask = (forecast['ds'] < df.ds.max())
    rmse = np.sqrt(np.mean((forecast[mask].yhat - df.y) ** 2))
    print('RMSE : {}'.format(rmse))


def charts(df, df_p, forecast):

    df_c = df.copy()
    df_c.set_index('ds', inplace = True)
    
    df_r = df.merge(forecast[['ds', 'yhat']], on = 'ds', how = 'inner')
    
    fig = plt.figure()
    fig, axs = plt.subplots(1, 3, figsize = (24, 5))

    axs[0].plot(df_c.index, df_c.y, label = 'y')
    axs[0].plot(forecast.ds, forecast.yhat, label = 'yhat'),
    axs[0].grid(True)
    axs[0].legend(loc = 'upper left', fancybox = True, prop = {'size' : 16})
    
    axs[1].plot(df_p.horizon.dt.days, df_p.rmse, label = 'RMSE')
    axs[1].set_xlabel('Prediction Horizon')
    axs[1].set_ylabel('RMSE')
    axs[1].grid(True) 
    axs[1].set_xlim([0, None])

    
    residuals = df_r.yhat - df_r.y
    
    axs[2].hist(residuals, bins = 20) 
    axs[2].set_xlabel('Residuals')
    axs[2].text(1000, 300, 'Skew = {:+.2f}'.format(skew(residuals)), fontsize = 14)
    axs[2].text(1000, 250, 'SD = {:.0f}'.format(np.std(residuals)), fontsize = 14)
    axs[2].axvline(0, color = '0.5')


def mape(y_true, y_pred):
    mape = np.mean(np.abs(y_true - y_pred) / y_true)
    return mape


def plot_forecast(df):
    fig = go.Figure(data = [
    go.Scatter(x = df.ds,y = df.y, name = 'y'),
    go.Scatter(x = forecast.ds, y = forecast.yhat, name='yhat'),
    go.Scatter(x = forecast['ds'], y = forecast['yhat_upper'], fill = 'tonexty', mode = 'none', name = 'upper'),
    go.Scatter(x = forecast['ds'], y = forecast['yhat_lower'], fill = 'tonexty', mode = 'none', name = 'lower'),
    go.Scatter(x = forecast['ds'], y = forecast['trend'], name='Trend')    
    ])

    fig.update_layout(
        autosize = False, 
        width = 800, 
        height = 400,
        margin = dict(
            l = 50,
            r = 50,
            b = 50,
            t = 50
        ))
    fig.show()
    

# def map(year):

#     tst_mask = (chart_df['event'] == 'leave_home') & (chart_df.index.year == year)
#     tst_df = chart_df[tst_mask]

#     fig = go.Figure(
#         data = [
#     #         Data for all rides based on date and time
#             Scattermapbox(
#                 lat = tst_df['lat'],
#                 lon = tst_df['lon'],
#                 mode = 'markers',
#                 marker = dict(
#                     showscale = True,
#     #                 color = np.append(np.insert(listCoords.index.hour, 0 , 0), 23),
#                     opacity = 0.5,
#                     size = 5,
#                     colorscale = [                
#                         [0, "#F4EC15"],
#                         [0.04167, "#DAF017"],
#                         [0.0833, "#BBEC19"],
#                         [0.125, "#9DE81B"],
#                         [0.1667, "#80E41D"],
#                         [0.2083, "#66E01F"],
#                         [0.25, "#4CDC20"],
#                         [0.292, "#34D822"],
#                         [0.333, "#24D249"],
#                         [0.375, "#25D042"],
#                         [0.4167, "#26CC58"],
#                         [0.4583, "#28C86D"],
#                         [0.50, "#29C481"],
#                         [0.54167, "#2AC093"],
#                         [0.5833, "#2BBCA4"],
#                         [1.0, "#613099"],
#                     ],
#     #                 colorbar = dict(
#     #                         title="Time of<br>Day",
#     #                         x=0.93,
#     #                         xpad=0,
#     #                         nticks=24,
#     #                         tickfont=dict(color="#d8d8d8"),
#     #                         titlefont=dict(color="#d8d8d8"),
#     #                         thicknessmode="pixels",                    
#     #                 ),
#                 ),
#             ),
#             # Plot of important locations on the map
#             Scattermapbox(
#                 lat=[list_of_locations[i]["lat"] for i in list_of_locations],
#                 lon=[list_of_locations[i]["lon"] for i in list_of_locations],
#                 mode="markers",
#                 hoverinfo="text",
#                 text=[i for i in list_of_locations],
#                 marker=dict(size=8, color="#ffa0a0"),
#             ),        
#         ],
#         layout=Layout(
#             autosize=True,
#             margin=go.layout.Margin(l=0, r=35, t=0, b=0),
#             showlegend=False,
#             mapbox=dict(
#                 accesstoken=mapbox_access_token,
#                 center=dict(lat=latInitial, lon=lonInitial),  # 40.7272  # -73.991251
#                 style="dark",
#                 bearing=bearing,
#                 zoom=zoom,
#             ),
#             updatemenus=[
#                 dict(
#                     buttons=(
#                         [
#                             dict(
#                                 args=[
#                                     {
#                                         "mapbox.zoom": 12,
#                                         "mapbox.center.lon": "-73.991251",
#                                         "mapbox.center.lat": "40.7272",
#                                         "mapbox.bearing": 0,
#                                         "mapbox.style": "dark",
#                                     }
#                                 ],
#                                 label="Reset Zoom",
#                                 method="relayout",
#                             )
#                         ]
#                     ),
#                     direction="left",
#                     pad={"r": 0, "t": 0, "b": 0, "l": 0},
#                     showactive=False,
#                     type="buttons",
#                     x=0.45,
#                     y=0.02,
#                     xanchor="left",
#                     yanchor="bottom",
#                     bgcolor="#323130",
#                     borderwidth=1,
#                     bordercolor="#6d6d6d",
#                     font=dict(color="#FFFFFF"),
#                 )
#             ],
#         ),
#     )
#     fig.update_layout(
#         autosize = False,
#         width = 1100,
#         height = 700
#     )

#     fig.show()