import matplotlib.pyplot as plt
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import vectorbt as vbt
import common_func as comf

def XYobj(index, y, name, color=None):
    if not color:
        obj = go.Scatter(x=index, y=y, mode='lines',name=name)
    else:
       obj = go.Scatter(x=index, y=y, mode='lines',name=name, line=dict(color=color))
    return obj

def Markerobj(std, mean_ret):
    obj = go.Scatter(x=std, y=mean_ret, mode='markers', marker=dict(size=5))
    return obj

def newPlotfunc(objDict):
    numPlots = len(objDict['plotList'])
    specs = []
    for plot in objDict['plotList']:
        specs.append([{'type': plot['type']}])

    fig = make_subplots(
        rows=numPlots, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.3,
        specs=specs,
    )

    for index, plotInfo in enumerate(objDict['plotList']):
        if plotInfo['type'] == 'scatter':
            for obj in plotInfo['obj']:
                fig.add_trace(obj, row=index+1, col=1)
            fig.update_xaxes(title_text = plotInfo['x'], row=index+1, col=1)
            fig.update_yaxes(title_text = plotInfo['y'], row=index+1, col=1)
        elif plotInfo['type'] == 'table':
            fig.add_trace(
                go.Table(
                    header=dict(
                        values=plotInfo['header'],
                        font=dict(size=10),
                        align='left'
                    ),
                    cells=dict(
                        values=plotInfo['data'],
                        align='left'
                    )
                ),
                row=index+1, col=1
            )

    layout = go.Layout(
        title=objDict['title'],
        xaxis=dict(rangeslider=dict(visible=True)),
        xaxis_fixedrange=False,  # Enable scrolling on the x-axis
        yaxis_fixedrange=False,  # Enable scrolling on the y-axis
        dragmode='pan',  # Enable panning
    )
    fig.update_layout(layout)
    fig.show()

def startPlot(param_dict):
    title = param_dict['title'] if 'title' in param_dict else None
    cumRetData = param_dict['cumRetData'] if 'cumRetData' in param_dict else None
    tabledf = param_dict['tabledf'] if 'tabledf' in param_dict else None
    data_index = param_dict['data_index'] if 'data_index' in param_dict else None

    objDict = {
        'title': title,
        'plotList': [
            # { # Scatter for Portfolio value
            #     'type': 'scatter',
            #     'x': 'Date',
            #     'y': 'Price (USD)',
            #     'obj': [plof.XYobj(sim_data_index, pf_rolling_val, 'Portfolio Value', color = 'Black')], # list of objects
            # },

            { # Scatter for all cumulative returns of stocks in portfolio
                'type': 'scatter',
                'x': 'Date',
                'y': 'Cumulative Returns',
                'obj': [XYobj(data_index, y, name, color = 'black' if name=='Optimized Portfolio' else None) for name,y in cumRetData.items()], # list of objects
            },

            { # Table for statistics per asset
                'type': 'table',
                'header': list(tabledf.columns),
                'data': [tabledf[col] for col in list(tabledf.columns)],
            },
        ],
    }

    newPlotfunc(objDict)

def monteCarloPlot(param_dict):
    title = param_dict['title'] if 'title' in param_dict else None
    MC_list = param_dict['MC_list'] if 'MC_list' in param_dict else None
    if MC_list is not None:
        MCreturn_list = MC_list[1]
        MCstd_list = MC_list[0]

    objDict = {
        'title': title,
        'plotList': [
            { # Monte Carlo Simulation - Efficient Frontier
                'type': 'scatter',
                'x': 'Annualized Volatility',
                'y': 'Expected Annualized Return',
                'obj': [Markerobj(MCstd_list, MCreturn_list)]
            },
        ],
    }
    newPlotfunc(objDict)