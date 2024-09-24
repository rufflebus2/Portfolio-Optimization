# Portfolio Optimization Project with Interactive Dashboard
This Portfolio Management project optimizes a portfolio with stocks of your choice. 

•	Conducted portfolio optimization with risk-return tradeoff analysis for stock selection, with Monte Carlo simulations for robustness and efficient frontier determination

•	Utilized Plotly for interactive dashboards illustrating individual stock and portfolio statistics (CAGR, Sharpe, Volatility) with comparison against equal-weighted portfolio

Python version: 3.11.5

# Instructions
1. **tickers**: List out your desired tickers
2. **start_date, end_date, interval**: Choose the data specifics: start and end dates to simulate over, granularity of data
3. **optType**: Choose the optimization method
4. **eq_weights**: Choose whether you want the stocks to be equally weighted or not (more information later)
5. **mcN**: Number of Portfolios to simulate to ensure robust optimization
6. Once done, run "python3 simulatePortfolio.py" in the terminal

# Optimization methods
Currently, there are only 2 optimization methods: min_var and opt_sharpe


**min_var** - Minimum Variance Portfolio: The optimizer conducts a Monte Carlo simulation to find the optimal weights for a minimum variance portfolio.

**opt_sharpe** - Optimal Sharpe Portfolio: The optimizer conducts a Monte Carlo simulation to find the optimal weights for an optimal sharpe portfolio.


# Equal weights variable

**eq_weights** - Boolean for equal weighting: Why would I need this if the min_var and opt_sharpe would determine the weights?

Good question. This variable is here explicitly to facilitate finding the optimal equal weighted portfolio according to optType when there are 
more stocks **(numAssets)** than the desired stock limit **(MAX_ASSETS)**. This allows the optimizer to choose the stocks
that maximizes sharpe/minimizes variance of an equally-weighted portfolio.

# Backtesting and Out-of-sample testing
The portfolios are optimized on the backtesting data, and the weights are used on out-of-sample data to measure effectiveness and robustness of the portfolio.
The **split_ratio** can be used to explicitly state how much % of the total data will be a part of backtesting.

# Data Visualization
Plotly was used to create the interactive dashboard, showing cumulative return graphs of the individual stocks in the portfolio, as well as the statistics 
(CAGR, Sharpe, Volatility, MDD).

