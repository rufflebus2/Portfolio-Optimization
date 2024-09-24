import pandas as pd
import numpy as np
import yfinance as yf
from scipy.optimize import minimize
import vectorbt as vbt
import os
import datetime as dt
import itertools
from multiprocessing import Pool

RF = 0.04
ANN_MULT = 252
MAX_ASSETS = 20

class CustomError(Exception):
    pass

def vecbtMetrics2():
    metrics = [
        'start',
        'end',
        'period',
        'start_value',
        'end_value',
        'total_return',
        'benchmark_return',
        'total_fees_paid',
        'max_dd',
        'max_dd_duration',
        # 'total_trades',
        # 'total_closed_trades',
        # 'total_open_trades',
        'win_rate',
        'sharpe_ratio',
        # 'best_trade',
        # 'worst_trade',
    ]
    return metrics

def save_data(tickers, start, end, interval, new_folder_path):
    count = 0
    for ticker in tickers:
        data = yf.download(ticker, start=start, end=end, interval=interval)
        if data.empty:
            continue
        else:
            csv_file_path = os.path.join(new_folder_path, f'{ticker}_data.csv')
            data.to_csv(csv_file_path) # Save to csv
            count += 1
    print(f'{count} datasets saved as CSV.')

def retrieve_data(tickers, start, end, interval, new_folder_path):
    folder_name = f'{start}to{end}'
    dataDict = {}
    nodataTickers = []
    for ticker in tickers:
        csv_file_path = os.path.join(new_folder_path, f'{ticker}_data.csv')

        try: 
            df = pd.read_csv(csv_file_path)
            df.set_index('Date', inplace=True)
            dataDict[ticker] = df
            # print(f'{ticker} retrieved for {folder_name}.')
        except FileNotFoundError:
            print(f'{ticker} does not exist for {folder_name}.')
            nodataTickers.append(ticker)
    
    if len(nodataTickers)!=0:
        # Handle recently added data
        save_data(nodataTickers, start, end, interval, new_folder_path)
        for ticker in nodataTickers:
            csv_file_path = os.path.join(new_folder_path, f'{ticker}_data.csv')
            df = pd.read_csv(csv_file_path)
            df.set_index('Date', inplace=True)
            dataDict[ticker] = df
    
    return dataDict

def prepare_data(tickers, start: str, end: str, interval):
    folder_name = f'{start}to{end}'
    parent = '/Users/raphaelbas/Desktop/vscode_workspace/manultrade_proj/portfolioManagement/csvdatafiles/'
    new_folder_path = os.path.join(parent, folder_name)
    folder_exist = False

    try:
        folder_exist = False
        os.mkdir(new_folder_path)
        print(f"Directory '{folder_name}' created successfully.")
    except FileExistsError:
        folder_exist = True
        print(f"Directory '{folder_name}' found.")

    if not folder_exist: # Retrieve from file
        save_data(tickers, start, end, interval, new_folder_path)
    dataDict = retrieve_data(tickers, start, end, interval, new_folder_path)
    return dataDict
    
def drop_keys(input_dict, keys_to_drop):
    empty_dict = {}
    for key, value in input_dict.items():
        if key not in keys_to_drop:
            empty_dict[key] = value
    return empty_dict

def drop_list(input_list, list_to_drop):
    empty_list = []
    for item in input_list:
        if item not in list_to_drop:
            empty_list.append(item)
    return empty_list

def valid_index(dataDict):
    beginning_index = None
    end_index = None
    for df in dataDict.values():
        begdate = dt.datetime.strptime(df.index[0], '%Y-%m-%d')
        enddate = dt.datetime.strptime(df.index[-1], '%Y-%m-%d')
        if beginning_index is None:
            beginning_index = df.index[0]
        elif begdate > dt.datetime.strptime(beginning_index, '%Y-%m-%d'):
            beginning_index = df.index[0]

        if end_index is None:
            end_index = df.index[-1]
        elif enddate < dt.datetime.strptime(end_index, '%Y-%m-%d'):
            end_index = df.index[-1]

    beginning_index = pd.to_datetime(beginning_index)
    end_index = pd.to_datetime(end_index)
    return beginning_index, end_index

def valid_data(dataDict, tickers): # Pass dataDict
    # df_len = 0
    # badTickerList = []
    # for ticker in tickers:
    #     dataDict[ticker].dropna(inplace=True)
    #     print(f'{ticker}:{len(dataDict[ticker])}')
    #     if df_len == 0:
    #         df_len = len(dataDict[ticker])
    #     elif df_len != len(dataDict[ticker]):
    #         badTickerList.append(ticker)
    # valid_dataDict = drop_keys(dataDict,badTickerList)
    # tickers = drop_list(tickers, badTickerList)
    # print(f'Deleted Tickers: {badTickerList}')
    # raise CustomError('Stopping code.')
    beginning_index, end_index = valid_index(dataDict)
    for ticker, df in dataDict.items():
        cols = df.columns
        df_copy = pd.DataFrame(df.values, columns=cols, index=pd.to_datetime(list(df.index)))
        dataDict[ticker] = df_copy[beginning_index:end_index]
    return dataDict, tickers

def split_data(valid_dataDict, tickers, split_ratio=0.5):
    backtest_dataDict = {}
    exec_dataDict = {}
    for ticker in tickers:
        split_index = int(len(valid_dataDict[ticker]) * split_ratio)
        backtest_dataDict[ticker] = valid_dataDict[ticker][:split_index]
        exec_dataDict[ticker] = valid_dataDict[ticker][split_index:]
    return backtest_dataDict, exec_dataDict
        
def create_dfs(dataVar, tickers=None): # dataVar can be a dictionary of dfs or single df (Leave tickers blank for single df)
    # dataVar can be backtestDataDict or execDataDict or pf_close
    if tickers is not None:
        price_df = pd.DataFrame({ticker: dataVar[ticker]['Adj Close'] for ticker in tickers})
    else:
        price_df = dataVar
    ret_df = price_df.pct_change(axis=0).fillna(0)
    logret_df = np.log1p(ret_df)
    cumret_df = np.cumprod(1 + ret_df)
    periodret_df = pd.DataFrame(cumret_df.iloc[-1]).T
    all_dfs = {
        'price_df': price_df,
        'ret_df': ret_df,
        'logret_df': logret_df,
        'cumret_df': cumret_df,
        'periodret_df': periodret_df,
    }
    return all_dfs

def append_dfs(all_dfs, new_dfs):
    for df_name, df in all_dfs.items():
        all_dfs[df_name] = pd.concat([df,new_dfs[df_name]],axis=1)
    return all_dfs

def getOptReqs(all_dfs):
    mean_ret = all_dfs['logret_df'].mean(axis=0)
    omega = np.cov(all_dfs['logret_df'].T)
    try:
        inv_omega = np.linalg.inv(omega)
    except:
        print('inv_omega does not exist.')
        inv_omega = None
    return mean_ret, omega, inv_omega

def updateDF(all_dfs, weights): # Update DF by dropping tickers with weight=0, Usually for exec data -> prepping for visualization
    allTickers = list(all_dfs['price_df'].columns)
    nonzero_ind = list(np.where(weights!=0)[0])
    zero_ind = list(np.where(weights==0)[0])
    pfTickers = [allTickers[i] for i in nonzero_ind]
    dropTickers = [allTickers[i] for i in zero_ind]
    updated_dfs = {}
    for df_type, df in all_dfs.items():
        updated_dfs[df_type] = df.copy().drop(columns=dropTickers)
    return pfTickers, updated_dfs

def getEqualWeights(weights):
    arr = np.zeros(len(weights))
    nonzero_ind = np.where(weights!=0)[0]
    count = float(len(nonzero_ind))
    arr[nonzero_ind] = 1.0
    arr /= count
    return arr

def getOptResults(optType, all_dfs, eq_weights=False, mcN=10000): # all_dfs = backtest_dataDict
    numAsset = len(list(all_dfs['price_df'].columns))
    # print(f'Number of assets: {numAsset}')
    mean_ret, omega, inv_omega = getOptReqs(all_dfs)

    def sumWeights(x): # eq
        return np.sum(x) - 1

    def numAssetsCons(x): # ineq
        return 20 - np.count_nonzero(x) 
    
    def validNonZero(x): # eq
        nonzero_ind = np.where(x != 0)[0]
        return int(np.all(x[nonzero_ind] >= 0.001)) - 1
    
    def compileCons(x):
        c1 = sumWeights(x) == 0
        c2 = numAssetsCons(x) >= 0
        c3 = validNonZero(x) == 0
        valid = c1 and c2 and c3
        return valid

    constraints = [
        {'type': 'eq', 'fun': sumWeights},
        {'type': 'ineq', 'fun': numAssetsCons},
        {'type': 'eq', 'fun': validNonZero},
    ]
    bounds = tuple((0, 1) for _ in range(len(mean_ret)))

    def calculate_portfolio_variance(weights):
        return np.dot(weights.T, np.dot(omega, weights))
    
    def negative_sharpe(weights):
        daily_rf = (1 + RF) ** (1 / ANN_MULT) - 1
        excess_ret = mean_ret - daily_rf
        mean_portfolio_return = np.dot(weights, excess_ret).mean()
        std_portfolio = excess_ret.std()
        return -(mean_portfolio_return-daily_rf) / std_portfolio
    
    def negative_mar(weights):
        pass

    opt_func = None
    if optType == 'min_var':
        opt_func = calculate_portfolio_variance
    elif optType == 'opt_sharpe':
        opt_func = negative_sharpe 
    
    func = np.inf
    func_weights = None
    backup = None
    if eq_weights:
        if numAsset > MAX_ASSETS:
            for i in range(mcN):
                weights = np.zeros(numAsset)
                random_indices = np.random.choice(numAsset, MAX_ASSETS, replace=False)
                weights[random_indices] = 1.0/MAX_ASSETS
                neg_func = calculate_portfolio_variance(weights) # negative_sharpe(weights)
                if neg_func < func:
                    func = neg_func
                    func_weights = weights
        else:
            func_weights = np.ones(numAsset)/numAsset
    else:
        for i in range(mcN):
            if numAsset > MAX_ASSETS:
                weights = np.zeros(numAsset)
                random_indices = np.random.choice(numAsset, MAX_ASSETS, replace=False)
                weights[random_indices] = np.random.random(MAX_ASSETS)
            else:
                weights = np.random.rand(numAsset)
            weights /= np.sum(weights)
            backup = weights
            if compileCons(weights):
                result = minimize(opt_func, weights, method='SLSQP', bounds = bounds, constraints=constraints)
                if result.fun < func and compileCons(result.x):
                    func = result.fun
                    func_weights = result.x
        
    if func_weights is None:
        if not eq_weights:
            print('ERROR: NO GOOD WEIGHTS. ASSIGNING EQUAL-WEIGHTED PORTFOLIO.')
        func_weights = getEqualWeights(backup)

    optimal_weights = func_weights
    optimal_return = float(np.dot(optimal_weights, mean_ret)) * ANN_MULT
    optimal_std = np.sqrt(np.dot(optimal_weights.T, np.dot(omega, optimal_weights))) * np.sqrt(ANN_MULT)
    # pfTickers, _ = updateDF(all_dfs, optimal_weights)
    # eq_weights = getEqualWeights(optimal_weights)

    optResults = {
        'return': optimal_return,
        'vol': optimal_std,
        'weights': optimal_weights,
        # 'eq weights': eq_weights,
        # 'pfTickers': pfTickers,
        # 'updated_dfs': updated_dfs, # dfs that dropped tickers
    }

    return optResults

def pfClose(price_df, weights, name):
    return pd.DataFrame({name: (price_df * weights).sum(axis=1)})

def pfDFs(all_dfs, weights, name = 'Optimized Portfolio'): # all_dfs = exec_dataDict 
    pf_close = pfClose(all_dfs['price_df'], weights, name)
    pf_dfs = create_dfs(pf_close)
    return pf_dfs

def dfForTable(optResults, all_dfs, pf_dfs): # all_dfs = exec_dataDict
    eq_weights = getEqualWeights(optResults['weights'])
    eq_dfs = pfDFs(all_dfs, eq_weights, 'Equally-Weighted Portfolio')
    pfTickers, updated_dfs = updateDF(all_dfs, optResults['weights'])
    updated_dfs = append_dfs(updated_dfs, pf_dfs)
    updated_dfs = append_dfs(updated_dfs, eq_dfs)
    return updated_dfs

def getSharpe(all_dfs):
    daily_rf = (1 + RF) ** (1 / ANN_MULT) - 1
    excess_returns = all_dfs['ret_df'] - daily_rf

    sharpe_ratio = excess_returns.mean(axis=0) / excess_returns.std(axis=0)
    annualized_sharpe_ratio = sharpe_ratio * np.sqrt(252)

    return annualized_sharpe_ratio

def getMDD(all_dfs):
    cum_ret = all_dfs['cumret_df']
    peak = np.maximum.accumulate(cum_ret)
    drawdowns = (cum_ret - peak)/peak 
    mdd = np.min(drawdowns,axis=0)
    return mdd

def getCAGR(all_dfs):
    final_cum_ret = all_dfs['periodret_df']
    period_length = len(all_dfs['price_df'])
    cagr = (1+final_cum_ret) ** (ANN_MULT/period_length) - 1
    return cagr

def getVol(all_dfs):
    returns = all_dfs['ret_df']
    ann_vol = returns.std() * np.sqrt(ANN_MULT)
    return ann_vol

def getMetrics(all_dfs): # all_dfs = exec_dataDict for metrics (usually)
    sharpe = getSharpe(all_dfs)
    mdd = getMDD(all_dfs)
    cagr = getCAGR(all_dfs)
    vol = getVol(all_dfs)
    return sharpe, mdd, cagr, vol

def round_stats(stat, dec = 2, perc=True):
    if perc:
        str_stat = np.round(stat*100,dec).astype(str)
        for i,x in enumerate(str_stat):
            str_stat[i] += '%'
        return str_stat
    else:
        return np.round(stat,2)

def getTableStats(updated_dfs, optResults=None): # updated_dfs = dfForTable(...)
    asset = list(updated_dfs['price_df'].columns)
    cumrets= round_stats(updated_dfs['periodret_df'].values.flatten()-1)
    cagr= round_stats(getCAGR(updated_dfs).values.flatten())
    mdd= round_stats(getMDD(updated_dfs).values.flatten())
    sharpe= round_stats(getSharpe(updated_dfs).values.flatten(), dec=4, perc=False)
    vol= round_stats(getVol(updated_dfs).values.flatten())

    nonzero_weights = None
    if optResults is not None:
        nonzero_ind = np.where(optResults['weights']!=0)[0]
        nonzero_weights = optResults['weights'][nonzero_ind]
        nonzero_weights = np.concatenate((nonzero_weights, np.zeros(2)))
        nonzero_weights = round_stats(nonzero_weights)

    # print('assets')
    # print(asset, len(asset))
    # print('cumrets')
    # print(cumrets, len(cumrets))
    # print('cagr')
    # print(cagr, len(cagr))
    # print('mdd')
    # print(mdd, len(mdd))
    # print('sharpe')
    # print(sharpe, len(sharpe))
    # print('vol')
    # print(vol, len(vol))
    # print('weights')
    # print(nonzero_weights, len(nonzero_weights))

    table = pd.DataFrame({
        'Asset': asset,
        'Cumulative return': cumrets,
        'CAGR': cagr,
        'MDD': mdd,
        'Annualized Sharpe': sharpe,
        'Annualized Volatility': vol,
        'Optimized Portfolio weighting': nonzero_weights,
    })
    # table.set_index('Asset', inplace=True)
    return table

def printWeights(optResults, updated_dfs):
    tickers = list(updated_dfs['price_df'].columns)
    nonzero_ind = np.where(optResults['weights']!=0)[0]
    nonzero_weights = optResults['weights'][nonzero_ind]
    weightDict = {tickers[i]: [f'{nonzero_weights[i]:.2%}'] for i in range(len(tickers)-2)}
    df = pd.DataFrame(weightDict)
    print(df)

def printResults(backtest_pf_dfs, exec_pf_dfs): # specified backtest_dataDict
    sharpe, mdd, cagr, vol = getMetrics(backtest_pf_dfs)
    print('------BACKTEST------')
    print(f'sharpe: {sharpe.values.flatten()[0]:.4f}')
    print(f'mdd: -{-mdd.values.flatten()[0]:.2%}')
    print(f'cagr: {cagr.values.flatten()[0]:.2%}')
    print(f'vol: {vol.values.flatten()[0]:.2%}')

    sharpe, mdd, cagr, vol = getMetrics(exec_pf_dfs)
    print('------EXEC------')
    print(f'sharpe: {sharpe.values.flatten()[0]:.4f}')
    print(f'mdd: -{-mdd.values.flatten()[0]:.2%}')
    print(f'cagr: {cagr.values.flatten()[0]:.2%}')
    print(f'vol: {vol.values.flatten()[0]:.2%}')

def getCumRet(updated_dfs):
    returnDict = {}
    cumret_df = updated_dfs['cumret_df']
    cols = list(cumret_df.columns)
    for ticker in cols:
        returnDict[ticker] = cumret_df[ticker].values
    return returnDict



