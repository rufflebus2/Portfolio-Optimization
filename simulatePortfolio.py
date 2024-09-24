import datetime as dt
import common_func as comf
import plotting_func as plof
import time

# DOWNLOAD DATA
finance = ['AMZN','AAPL', 'META','MSFT', 'GOOGL','NVDA',]
tech = ['AMZN','AAPL', 'META','MSFT', 'GOOGL','NVDA',]
health = ['UNH','LLY','JNJ','ABBV', 'MRK','PFE',]
energy = ['XOM','CVX','SHEL','TTE',]
real_estate = ['PLD','AMT','EQIX','WELL','PSA',]
cons_def = ['WMT','PG','COST','KO','PEP',]
industrial = ['GE','CAT','RTX','UNP','LMT',]
v_mut_funds = [
    'VIGAX', # Growth Index Fund Admiral Shares
    'VWILX', # International Growth Fund Admiral Shares
    'VSPGX', # SP500 Growth Index Fund Institutional Shares
    'VWUAX', # US Growth Fund Admiral Shares
    'VEIRX', # Equity Income Fund Admiral Shares
    'VCOBX', # Core Bond Fund Admiral Shares
    'VFIDX', # Intermediate-Term Investment Grade Fund Admiral Shares
    'VWETX', # Long-Term Investment Grade Fund Admiral Shares
    'VFSUX', # Short-Term Investment Grade Fund Admiral Shares
]
tickers = v_mut_funds + tech + finance + health + energy + real_estate + cons_def + industrial

split_ratio = 0.5 # 0.2 = 20% backtest
end_date = '2024-01-01' # dt.datetime.now().strftime('%Y-%m-%d')
start_date = '2000-01-01' # (end - dt.timedelta(days = 8*365)).strftime('%Y-%m-%d')
interval = '1d'
optType = 'opt_sharpe' # opt_sharpe, min_var
eq_weights = True
mcN = 50000

# Prepare data
start = time.time()
dataDict = comf.prepare_data(tickers, start_date, end_date, interval)
valid_dataDict, tickers = comf.valid_data(dataDict, tickers)
backtest_dataDict, exec_dataDict = comf.split_data(valid_dataDict, tickers, split_ratio)

# Get dfs
backtest_dfs = comf.create_dfs(backtest_dataDict, tickers)
exec_dfs = comf.create_dfs(exec_dataDict, tickers)
end = time.time()
print(f'Data preparation ({end-start:.2f}s)')

# Optimization
print('Optimization start')
start = time.time()
optResults = comf.getOptResults(optType, backtest_dfs, eq_weights=eq_weights, mcN=mcN) # Customize mcN here
exec_pf_dfs = comf.pfDFs(exec_dfs, optResults['weights'])
backtest_pf_dfs = comf.pfDFs(backtest_dfs, optResults['weights'])
end = time.time()
print(f'Optimization end ({end-start:.2f}s)')

# Prepare stats for visuals
start = time.time()
updated_dfs = comf.dfForTable(optResults, exec_dfs, exec_pf_dfs) # backtest_dfs, backtest_pf_dfs # exec_dfs, exec_pf_dfs
table = comf.getTableStats(updated_dfs, optResults)
cumRetData = comf.getCumRet(updated_dfs)
end = time.time()
print(f'Visualization preparation({end-start:.2f}s)')

comf.printResults(backtest_pf_dfs, exec_pf_dfs)
comf.printWeights(optResults, updated_dfs)

plotParam_dict = {
    'title': f'{optType} Portfolio',
    'cumRetData': cumRetData,
    'tabledf': table,
    'data_index': exec_dfs['price_df'].index, # change this for backtest visualization
}

# Plotting 
plotting = True # Change this if needed
if plotting:
    plof.startPlot(plotParam_dict)