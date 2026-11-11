#!/usr/bin/env python
# coding: utf-8

# Install Alpaca Trade API to get the data
# 

# In[ ]:


pip install alpaca_trade_api


# Install deap for Genetic Algorithm Optimisation

# In[ ]:


pip install deap


# In[ ]:


import os
import pandas as pd
import alpaca_trade_api as tradeapi
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from sklearn.linear_model import LinearRegression
from deap import base, creator, tools, algorithms
import random


# Function to Fetch The Data

# In[ ]:


def fetch_data(ticker,timeframe, start_date, end_date,alpaca_api_key,alpaca_secret_key):
  alpaca = tradeapi.REST(alpaca_api_key, alpaca_secret_key, api_version="v2")


  df_ticker = alpaca.get_bars(
      ticker,
      timeframe,
      start=start_date,
      end=end_date,
      limit=50000
  ).df

  df_ticker.reset_index(inplace=True)
  data = df_ticker

  stock = ticker[1]
  etf = ticker[0]
  stock_df = pd.DataFrame()
  etf_df = pd.DataFrame()
  stock_df[stock] = data[data['symbol'] == stock].set_index('timestamp')['close']
  etf_df[etf] = data[data['symbol'] == etf].set_index('timestamp')['close']

  df = pd.merge(etf_df[[etf]], stock_df[[stock]], left_index=True, right_index=True, how='outer')
  #df.fillna(method='bfill', inplace=True)
  df.bfill(inplace=True)
  return df


# \begin{equation}
# \text{Total Return} = \frac{P_{\text{end}} - P_{\text{start}}}{P_{\text{start}}}
# \end{equation}

# \begin{equation}
# \text{APR} = \left(1 + R\right)^{\frac{252}{n}} - 1
# \end{equation}
# 

# \begin{equation}
# \text{Sharpe Ratio} = \frac{\sqrt{252} \cdot \mathbb{E}[R]}{\sigma_R}
# \end{equation}
# 

# \begin{equation}
# \text{Maximum Drawdown} = \frac{\text{Trough Value} - \text{Peak Value}}{\text{Peak Value}}
# \end{equation}

# \begin{equation}
# \text{Max Drawdown Duration} = \max \left( \text{Days}_{\text{peak to trough}} \right)
# \end{equation}
# 

# In[ ]:


# Metrics calculation
def calculate_metrics(cumret):
    total_return = (cumret[-1] - cumret[0]) / cumret[0]
    apr = (1 + total_return) ** (252 / len(cumret)) - 1
    rets = pd.DataFrame(cumret).pct_change()
    sharpe = np.sqrt(252) * np.nanmean(rets) / np.nanstd(rets)

    highwatermark = np.zeros(cumret.shape)
    drawdown = np.zeros(cumret.shape)
    drawdownduration = np.zeros(cumret.shape)
    for t in np.arange(1, cumret.shape[0]):
        highwatermark[t] = np.maximum(highwatermark[t-1], cumret[t])
        drawdown[t] = cumret[t] / highwatermark[t] - 1
        if drawdown[t] == 0:
            drawdownduration[t] = 0
        else:
            drawdownduration[t] = drawdownduration[t-1] + 1
    maxDD = np.min(drawdown)
    maxDDD = np.max(drawdownduration)

    return total_return, apr, sharpe, maxDD, maxDDD


# In[ ]:


start_date = pd.Timestamp("2022-08-15", tz="America/New_York").isoformat()
end_date = pd.Timestamp("2023-09-15", tz="America/New_York").isoformat()
stock='JPM'
etf='XLF'


# ### Trend Following Backtesting

# Fetching the data for 15 minute, 30 minute and 1 hour time frame

# In[ ]:


df_15m=fetch_data([etf, stock],'15min',start_date, end_date,"PKYZ9BYQT638RSA9YAAC","XK19x09SvrAw9I32gpM2xYqVgNTuwJbEt1sstd3k")


# In[ ]:


df_30m=fetch_data([etf, stock],'30min',start_date, end_date,"PKYZ9BYQT638RSA9YAAC","XK19x09SvrAw9I32gpM2xYqVgNTuwJbEt1sstd3k")


# In[ ]:


df_1h=fetch_data([etf, stock],'1H',start_date, end_date,"PKYZ9BYQT638RSA9YAAC","XK19x09SvrAw9I32gpM2xYqVgNTuwJbEt1sstd3k")


# \begin{equation}
# \text{MACD Line} = \text{EMA}_{long_window} - \text{EMA}_{short_window}
# \end{equation}
# 
# \begin{equation}
# \text{Signal Line} = \text{EMA}_{signal_window}(\text{MACD Line})
# \end{equation}
# 
# \begin{equation}
# \text{MACD Histogram} = \text{MACD Line} - \text{Signal Line}
# \end{equation}
# 

# \begin{equation}
# \text{Average Gain} = \frac{\sum_{i=1}^{rsi_window} \text{Gain}_i}{rsi_window}
# \end{equation}
# 
# \begin{equation}
# \text{Average Loss} = \frac{\sum_{i=1}^{rsi_window} \text{Loss}_i}{rsi_window}
# \end{equation}
# 
# \begin{equation}
# \text{RS} = \frac{\text{Average Gain}}{\text{Average Loss}}
# \end{equation}
# 
# \begin{equation}
# \text{RSI} = 100 - \frac{100}{1 + \text{RS}}
# \end{equation}
# 

# 
# \begin{equation}
# \text{EMA}_t = \alpha \cdot P_t + (1 - \alpha) \cdot \text{EMA}_{t-1}
# \end{equation}
# 
# \begin{equation}
# \alpha = \frac{2}{19 + 1}
# \end{equation}
# 
# \begin{equation}
# \text{EMA}_0 = \frac{1}{ema_window} \sum_{i=1}^{ema_window} P_i
# \end{equation}
# 
# 
# 

# ##Genetic Algorithm Function Definition
# 
# ###Initialization:
# 
# Population: Create an initial population of individuals, where each individual represents a possible set of parameters for the trading strategy (e.g., short_window, long_window, signal_window, rsi_window, ema_window).
# 
# ###Evaluation:
# 
# Fitness Function: Each individual is evaluated using the fitness function, which in this case calculates the total return of the trading strategy using the individual’s parameters. The goal is to maximize this return.
# 
# ###Selection:
# 
# Select Parents: Choose individuals from the population based on their fitness scores. Individuals with higher fitness have a higher chance of being selected as parents.
# 
# ###Crossover (Recombination):
# 
# Combine Parents: Apply crossover operations to pairs of selected parents to create offspring. This involves combining parts of each parent's parameters to generate new individuals.
# 
# ###Mutation:
# 
# Alter Offspring: Apply mutation operations to offspring with a certain probability. This involves making random changes to some parameters to introduce variability and explore new parts of the solution space.
# 
# ###Replacement:
# 
# New Generation: Replace some or all of the old population with the new offspring to form the next generation.
# 
# ###Iteration:
# 
# Repeat: Repeat the evaluation, selection, crossover, mutation, and replacement steps for a specified number of generations (iterations).
# 
# ###Output:
# 
# Best Solution: After the final generation, the algorithm outputs the best individual found, which represents the optimal set of parameters for the trading strategy.
# 
# ###Summary
# The genetic algorithm iteratively evolves a population of possible solutions by simulating natural selection. It combines and mutates parameters to improve the trading strategy's performance, eventually converging on an optimal set of parameters that maximize the strategy's total return.
# 
# ## Paper - https://www.jstor.org/stable/24939139?seq=1

# In[ ]:


def optimize_trend_following_params(df, stock, population_size=50, generations=5):
    # Define the problem as a maximization problem
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    # Define the individual and population
    toolbox = base.Toolbox()
    toolbox.register("attr_int", random.randint, 10, 100)  # Adjust ranges as needed
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, 5)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Define the evaluation function
    def evaluate_strategy(individual):
        short_window, long_window, signal_window, rsi_window, ema_window = individual

        # Apply your indicators with the parameters from the individual
        df_macd = calculate_macd(df.copy(), short_window, long_window, signal_window)
        df_rsi = calculate_rsi(df_macd.copy(), rsi_window)
        df_final = calculate_ema(df_rsi.copy(), ema_window)

        # Apply the strategy
        _, _, _, _, _, rolling_cap = apply_trend_following_strategy(
        df_final,
        stock,
        window_size=30,
        short_window=individual[0],
        long_window=individual[1],
        signal_window=individual[2],
        rsi_window=individual[3],
        ema_window=individual[4]
    )

        # Calculate total return from rolling_cap
        cumret = pd.Series(rolling_cap).values
        total_return, _, _, _, _ = calculate_metrics(cumret)

        return -total_return,  # Return a tuple

    # Register the genetic algorithm operators
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutUniformInt, low=5, up=50, indpb=0.2)  # Adjust ranges as needed
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", evaluate_strategy)

    # Initialize the population
    population = toolbox.population(n=population_size)

    # Run the genetic algorithm
    algorithms.eaSimple(population, toolbox, cxpb=0.7, mutpb=0.2, ngen=generations, verbose=True)

    # Extract the best individual
    best_individual = tools.selBest(population, k=1)[0]

    # Return the parameters directly
    return tuple(best_individual)


# In[ ]:


# Calculate MACD
def calculate_macd(df, short_window, long_window, signal_window):
    df['MACD'] = df[stock].ewm(span=short_window, adjust=False).mean() - df[stock].ewm(span=long_window, adjust=False).mean()
    df['MACD_signal'] = df['MACD'].ewm(span=signal_window, adjust=False).mean()
    df['MACD_hist'] = df['MACD'] - df['MACD_signal']
    return df

# Calculate RSI
def calculate_rsi(df, rsi_window):
    delta = df[stock].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_window).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df

# Calculate EMA
def calculate_ema(df, ema_window):
    df['EMA'] = df[stock].ewm(span=ema_window, adjust=False).mean()
    return df


# Trend Following Strategy Function to return the Trade Details (of the best performing time frame) like Entry Price, Exit Price, Entry Exit Time

# In[ ]:


def apply_trend_following_strategy(df, stock, window_size, short_window, long_window, signal_window, rsi_window, ema_window,initial_capital=1000, commission=0.000119):

  df = calculate_macd(df, short_window, long_window, signal_window)
  df = calculate_rsi(df,rsi_window)
  df = calculate_ema(df,ema_window)

  # Parameters
  cap = float(1000)  # $1000
  commission = 0.000119
  win = 30
  quantity = []
  entry_price = []
  entry_time = []
  exit_time = []
  exit_price = []
  profits = []
  rolling_cap = [cap]

  open_pos = False
  position = 0

  for i in range(win, len(df)):

      df_win = df.iloc[i-win:i].copy()
      df_win.dropna(inplace=True)

      if len(df_win) < win:
          continue

    # Signals
      macd_signal = df_win['MACD'].iloc[-1]
      macd_hist = df_win['MACD_hist'].iloc[-1]
      rsi = df_win['RSI'].iloc[-1]
      price = df_win[stock].iloc[-1]
      #sma = df_win['SMA'].iloc[-1]
      sma = df_win['EMA'].iloc[-1]

      # Trend-following strategy
      if macd_hist > 0 and df_win[stock].iloc[-1] > sma and rsi < 70:
          if not open_pos:
              open_pos = True
              position = 1  # Long position
              entry_price.append(price)
              entry_time.append(df_win.index[-1])
              quantity.append(rolling_cap[-1] / price)

      elif macd_hist < 0 and df_win[stock].iloc[-1] < sma and rsi > 30:
          if not open_pos:
              open_pos = True
              position = -1  # Short position
              entry_price.append(price)
              entry_time.append(df_win.index[-1])
              quantity.append(rolling_cap[-1] / price)

      if open_pos:
          if (position == 1 and (macd_hist < 0 or rsi > 70)) or (position == -1 and (macd_hist > 0 or rsi < 30)):
              open_pos = False
              exit_price.append(price)
              exit_time.append(df_win.index[-1])
              if position == 1:
                  profits.append(price - entry_price[-1])
              else:
                  profits.append(entry_price[-1] - price)
              rolling_cap.append(rolling_cap[-1] + (profits[-1] - commission) * quantity[-1])
              position = 0
  return entry_price, entry_time, exit_time, exit_price, profits, rolling_cap


# In[ ]:


import pandas as pd

def apply_trend_following_strategy_monte_carlo(df, window_size, short_window, long_window, signal_window, rsi_window, ema_window, initial_capital=1000, commission=0.000119):

    def calculate_macd(df, short_window, long_window, signal_window):
        df['EMA12'] = df[0].ewm(span=short_window, adjust=False).mean()
        df['EMA26'] = df[0].ewm(span=long_window, adjust=False).mean()
        df['MACD'] = df['EMA12'] - df['EMA26']
        df['Signal Line'] = df['MACD'].ewm(span=signal_window, adjust=False).mean()
        df['MACD_hist'] = df['MACD'] - df['Signal Line']
        return df

    def calculate_rsi(df, period=rsi_window):
        delta = df[0].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        return df

    def calculate_ema(df, span=ema_window):
        df['EMA'] = df[0].ewm(span=span, adjust=False).mean()
        return df

    results = {}

    for stock in df.columns:
        df_stock = df[[stock]].copy()
        df_stock.columns = [0]  # Rename the column to 0 to avoid issues with DataFrame operations
        df_stock[0] = pd.to_numeric(df_stock[0], errors='coerce')
        df_stock = df_stock.ffill()
        df_stock = calculate_macd(df_stock, short_window, long_window, signal_window)
        df_stock = calculate_rsi(df_stock, rsi_window)
        df_stock = calculate_ema(df_stock, ema_window)

        cap = initial_capital
        quantity = []
        entry_price = []
        entry_time = []
        exit_time = []
        exit_price = []
        profits = []
        rolling_cap = [cap]

        open_pos = False
        position = 0

        for i in range(window_size, len(df_stock)):

            df_win = df_stock.iloc[i-window_size:i].copy().dropna()

            if len(df_win) < window_size:
                continue

            macd_signal = df_win['MACD'].iloc[-1]
            macd_hist = df_win['MACD_hist'].iloc[-1]
            rsi = df_win['RSI'].iloc[-1]
            price = df_win[0].iloc[-1]
            sma = df_win['EMA'].iloc[-1]

            if macd_hist > 0 and price > sma and rsi < 70:
                if not open_pos:
                    open_pos = True
                    position = 1  # Long position
                    entry_price.append(price)
                    entry_time.append(df_win.index[-1])
                    quantity.append(rolling_cap[-1] / price)

            elif macd_hist < 0 and price < sma and rsi > 30:
                if not open_pos:
                    open_pos = True
                    position = -1  # Short position
                    entry_price.append(price)
                    entry_time.append(df_win.index[-1])
                    quantity.append(rolling_cap[-1] / price)

            if open_pos:
                if (position == 1 and (macd_hist < 0 or rsi > 70)) or (position == -1 and (macd_hist > 0 or rsi < 30)):
                    open_pos = False
                    exit_price.append(price)
                    exit_time.append(df_win.index[-1])
                    if position == 1:
                        profits.append(price - entry_price[-1])
                    else:
                        profits.append(entry_price[-1] - price)
                    rolling_cap.append(rolling_cap[-1] + (profits[-1] - commission) * quantity[-1])
                    position = 0

        # Store results for the current stock
        results[stock] = {
            'entry_price': entry_price,
            'entry_time': entry_time,
            'exit_time': exit_time,
            'exit_price': exit_price,
            'profits': profits,
            'rolling_cap': rolling_cap
        }

    return results


# Trend Following Strategy Function to return the Rolling Capital Only (for calculating metrices for each dataset of different time frames)

# In[ ]:


def apply_trend_following_strategy_rc(df, stock, window_size, short_window, long_window, signal_window, rsi_window, ema_window, initial_capital=1000, commission=0.000119):

    df = calculate_macd(df, short_window, long_window, signal_window)
    df = calculate_rsi(df,rsi_window)
    df = calculate_ema(df,ema_window)

    cap = initial_capital
    quantity = []
    entry_price = []
    entry_time = []
    exit_time = []
    exit_price = []
    profits = []
    rolling_cap = [cap]

    open_pos = False
    position = 0

    for i in range(window_size, len(df)):
        df_win = df.iloc[i-window_size:i].copy()
        df_win.dropna(inplace=True)

        if len(df_win) < window_size:
            continue

        # Signals
        macd_hist = df_win['MACD_hist'].iloc[-1]
        rsi = df_win['RSI'].iloc[-1]
        price = df_win[stock].iloc[-1]
        sma = df_win['EMA'].iloc[-1]

        # Trend-following strategy
        if macd_hist > 0 and df_win[stock].iloc[-1] > sma and rsi < 70:
            if not open_pos:
                open_pos = True
                position = 1  # Long position
                entry_price.append(price)
                entry_time.append(df_win.index[-1])
                quantity.append(rolling_cap[-1] / price)

        elif macd_hist < 0 and df_win[stock].iloc[-1] < sma and rsi > 30:
            if not open_pos:
                open_pos = True
                position = -1  # Short position
                entry_price.append(price)
                entry_time.append(df_win.index[-1])
                quantity.append(rolling_cap[-1] / price)

        if open_pos:
            if (position == 1 and (macd_hist < 0 or rsi > 70)) or (position == -1 and (macd_hist > 0 or rsi < 30)):
                open_pos = False
                exit_price.append(price)
                exit_time.append(df_win.index[-1])
                if position == 1:
                    profits.append(price - entry_price[-1])
                else:
                    profits.append(entry_price[-1] - price)
                rolling_cap.append(rolling_cap[-1] + (profits[-1] - commission) * quantity[-1])
                position = 0

    return rolling_cap


# In[ ]:


#short_window, long_window, signal_window, rsi_window, ema_window = optimize_trend_following_params(df_1h, 'JPM')


# In[ ]:


short_window, long_window, signal_window, rsi_window, ema_window = 69,48,55,81,19


# In[ ]:


print(short_window, long_window, signal_window, rsi_window, ema_window)


# Print the Metrices for Trend Following Strategy

# In[ ]:


def print_metrics_trend(df):
  rolling_cap = apply_trend_following_strategy_rc(df, 'JPM', window_size=30, short_window=short_window, long_window=long_window, signal_window=signal_window, rsi_window=rsi_window, ema_window=ema_window)
  total_return, apr, sharpe, maxDD, maxDDD = calculate_metrics(np.array(rolling_cap))
  print(f"Total Return: {total_return * 100} %")
  print(f"APR: {apr * 100} %")
  print(f"Sharpe Ratio: {sharpe}")
  print(f"Max Drawdown: {maxDD}")


# Function to Draw the Buy & Hold vs Strategy Plot of Cummulative Pnl

# In[ ]:


def plot(df_pnl,df):

 fig, ax1 = plt.subplots()

 # plot cumulative PnL on the left y-axis
 ax1.plot(df_pnl['Exit_time'], df_pnl['pnl'].cumsum(), color='b', label='Cumulative PnL')
 ax1.set_xlabel('Exit Time')
 ax1.set_ylabel('Cumulative PnL', color='b')
 ax1.tick_params(axis='y', labelcolor='b')

 #plot buy and hold
 ax2 = ax1.twinx()
 buy_hold = (df[stock] - df[stock][0]) / df[stock][0]

 ax2.plot(df.index, buy_hold, color='r')

 ax2.set_ylabel('Buy & Hold Returns', color='r')
 ax2.tick_params(axis='y', labelcolor='r')

 lines, labels = ax1.get_legend_handles_labels()
 lines2, labels2 = ax2.get_legend_handles_labels()
 ax1.legend(lines + lines2, labels + labels2, loc='upper left')

 plt.title('Cumulative PnL vs. Buy & Hold Returns')
 plt.grid(True)
 plt.show()

 ax = sns.kdeplot(data=df_pnl['Rolling Capital'].pct_change())
 plt.show()


# Function to Create the Trades Details (like entry_time, entry_price, exit_time, exit_price, profits, rolling_cap)  Dataframe

# In[ ]:


def trades_details(entry_time, entry_price, exit_time, exit_price, profits, rolling_cap):

 min_length = min(len(entry_time), len(entry_price), len(exit_time),
                 len(exit_price), len(profits), len(rolling_cap[1:]))

 df_pnl = pd.DataFrame({
     "Entry_time": entry_time[:min_length],
     "Entry_price": entry_price[:min_length],
     "Exit_time": exit_time[:min_length],
     "Exit_price": exit_price[:min_length],
     'pnl': profits[:min_length],
     'Rolling Capital': rolling_cap[1:min_length+1]  # +1 to keep the slice consistent
      })
 return df_pnl


# In[ ]:


print_metrics_trend(df_15m)


# In[ ]:


print_metrics_trend(df_30m)


# In[ ]:


print_metrics_trend(df_1h)


# Since we get the best results for 1-hour time frame so we will proceed with 1-hour time frame for our Trend Following Strategy

# In[ ]:


entry_time, entry_price, exit_time, exit_price, profits, rolling_cap = apply_trend_following_strategy(df_1h,stock,30,short_window, long_window, signal_window, rsi_window, ema_window)
df_pnl = trades_details(entry_time, entry_price, exit_time, exit_price, profits, rolling_cap)
plot(df_pnl,df_1h)


# Trade Details for the Trend Following Strategy

# In[ ]:


df_pnl


# ### Setup and Initialization:
# dt = 1 / days: This sets the time increment for each step in the simulation.
# N = days + 1: This defines the total number of time steps (including the initial point).
# t = np.linspace(0, 1, N): This generates an array of time points from 0 to 1, split into N intervals. This represents the normalized time scale of the simulation.

# ###Simulate the Wiener Process (Brownian Motion):
# W = np.random.standard_normal(size=N): This generates N standard normal random variables, representing increments of the Wiener process (or Brownian motion).
# W = np.cumsum(W) * np.sqrt(dt): This computes the cumulative sum of the increments to simulate the Brownian motion and scales it by square root(dt)
#  . This produces the path of the Wiener process over time.
# 
# \begin{equation}
# W_t = \sum_{i=1}^{N} \epsilon_i \sqrt{\Delta t}
# \end{equation}

# ### Simulate the Stock Price Path:
# 
# X: This computes the log of the stock price according to the GBM model. Here:
# mu is the drift term (expected return).
# sigma is the volatility.
# sigma * W represents the stochastic component driven by the Brownian motion.
# 
# \begin{equation}
# X_t = (\mu - 0.5 \sigma^2) t + \sigma W_t
# \end{equation}

# ###Compute the Stock Prices:
# 
# S: This converts the log prices back to the actual stock prices by exponentiating X and scaling by the initial price. This represents the simulated stock price path over time.
# 
# \begin{equation}
# S_t = S_0 \exp(X_t)
# \end{equation}

# In[ ]:


# Function to simulate stock returns
def simulate_returns(initial_price, mu, sigma, days):
    dt = 1 / days
    N = days + 1
    t = np.linspace(0, 1, N)

    W = np.random.standard_normal(size=N)
    W = np.cumsum(W) * np.sqrt(dt)

    X = (mu - 0.5 * sigma**2) * t + sigma * W
    S = initial_price * np.exp(X)
    price = pd.Series(S)
    returns = price.pct_change().dropna()

    return pd.Series(returns)


# ### Mean Reversion Backtesting

# The apply_mean_reversion_strategy function implements a mean reversion strategy for trading a stock and its corresponding ETF. Mean reversion is a financial theory suggesting that asset prices and returns eventually revert to the long-term mean or average level of the entire dataset.
# 
# Here’s a step-by-step explanation of the strategy:
# 
# Initial Setup:
# 
# Parameters:
# df: The dataframe containing the stock and ETF prices.
# stock and etf: The specific stock and ETF being analyzed.
# window_size: The rolling window size for analysis (set to 30 in the code).
# initial_capital: The initial amount of money available for trading (defaulted to $1000).
# commission: The trading commission cost (set as 0.000119).
# Rolling Window Analysis:
# 
# For each window of 30 days (or the specified window_size), the function calculates the returns of both the stock and ETF.
# These returns are then standardized by subtracting the mean and dividing by the standard deviation.
# Linear Regression:
# 
# A linear regression is performed between the standardized returns of the ETF (independent variable
# 𝑋
# X) and the stock (dependent variable
# 𝑦
# y) to calculate the residual
# 𝜖
# ϵ.
# This residual is further analyzed to detect mean-reverting behavior by using a second regression to estimate the Ornstein-Uhlenbeck (OU) process parameters.
# Ornstein-Uhlenbeck (OU) Process:
# 
# The OU process models the stock's mean-reverting behavior. The key parameters are:
# Kappa (
# 𝜅
# κ): The rate of mean reversion.
# M (
# 𝑚
# m): The mean level to which the process reverts.
# Sigma (
# 𝜎
# σ): The volatility of the process.
# Zeta (
# 𝜁
# ζ): The noise or error term.
# The strategy computes a standardized score
# 𝑠
# s (often referred to as the "z-score") which helps in determining whether the stock is overbought or oversold relative to the ETF.
# Trading Logic:
# 
# Entry: If the z-score exceeds a certain threshold (e.g.,
# 1.25
# 1.25 or
# −
# 1.25
# −1.25), a trading position is opened, betting that the price will revert to the mean.
# Exit: The position is closed if the z-score reverts closer to zero (indicating mean reversion) or if it crosses a stop-loss threshold (e.g.,
# 1.5
# 1.5 or
# −
# 1.5
# −1.5).
# Profit Calculation: Profits are calculated based on the difference between the entry and exit prices, adjusted for commission and quantity traded.
# Capital and Performance Tracking:
# 
# The strategy tracks the rolling capital over time, adjusting it based on profits and losses from trades.

# In[ ]:


def level_crosses(series, level):  #double check
        series=series.values
        change = 0
        if series[1]>level and series[0]<level:
            change=1
        elif series[1]<level and series[0]>level:
            change=-1
        else:
            change=0
        return change

def single_level_crosses(series, level):  #single check
    series=series.values
    change = 0
    if series[0]>level:
        change=-1
    elif series[0]<level:
        change=1
    else:
        change=0
    return change


# # Mean Reversion Strategy: Detailed Explanation & Key Formulas
# 
# ### Introduction
# This strategy is based on the idea that prices tend to revert to their mean over time. The approach uses a combination of linear regression and an Ornstein-Uhlenbeck process to identify when a stock's price deviates significantly from its mean and is likely to revert.
# 
# ### 1. Returns Calculation
# To normalize price data, we first calculate the return \( R_t \) over a window of time:
# 
# \begin{equation}
# R_t = \frac{P_t - P_{t-1}}{P_{t-1}}
# \end{equation}
# 
# Where \( P_t \) is the price at time \( t \).
# 
# ### 2. Standardization
# The returns are then standardized to obtain a z-score:
# 
# \begin{equation}
# Z_t = \frac{R_t - \mu_R}{\sigma_R}
# \end{equation}
# 
# Here, \( \mu_R \) and \( \sigma_R \) represent the mean and standard deviation of returns, respectively.
# 
# ### 3. Linear Regression
# A linear regression model is applied between the stock and ETF returns:
# 
# \begin{equation}
# y_t = \beta_0 + \beta_1 X_t + \epsilon_t
# \end{equation}
# 
# Where \( y_t \) is the stock return, \( X_t \) is the ETF return, and \( \epsilon_t \) is the residual.
# 
# ### 4. Ornstein-Uhlenbeck Process
# The Ornstein-Uhlenbeck (OU) process is a continuous-time stochastic process often used to model mean-reverting behavior. The parameters of the OU process, such as the mean reversion speed (
# 𝜅
# κ), are derived from the AR(1) process. And we use the residuals from the linear regression (epsilon) to form the AR(1) process.
# The residuals from the regression are used to model the mean-reverting process:
# 
# \begin{equation}
# dX_t = \kappa (\mu - X_t) dt + \sigma dW_t
# \end{equation}
# 
# Where \( \kappa \) is the speed of mean reversion, \( \mu \) is the long-term mean, and \( \sigma \) is the volatility.
# 
# ### 5. Z-Score (s-score)
# The z-score is calculated to assess the deviation from the mean:
# 
# \begin{equation}
# s_t = \frac{X_t - \mu}{\sigma_{\text{eq}}}
# \end{equation}
# 
# ### 6. Trade Entry and Exit Conditions
# 
# - **Entry Condition (Upper Threshold):**
# 
# \begin{equation}
# s_t \geq 1.25
# \end{equation}
# 
# - **Entry Condition (Lower Threshold):**
# 
# \begin{equation}
# s_t \leq -1.25
# \end{equation}
# 
# - **Exit Condition (Upper Target):**
# 
# \begin{equation}
# s_t \leq 0.5
# \end{equation}
# 
# - **Exit Condition (Lower Target):**
# 
# \begin{equation}
# s_t \geq -0.5
# \end{equation}
# 
# - **Stop Loss (Upper):**
# 
# \begin{equation}
# s_t \geq 1.5
# \end{equation}
# 
# - **Stop Loss (Lower):**
# 
# \begin{equation}
# s_t \leq -1.5
# \end{equation}
# 
# ### 7. Profit Calculation
# Finally, the profit from each trade is calculated as:
# 
# \begin{equation}
# \text{Profit}_t = Q \times (P_{\text{entry}} - P_{\text{exit}} - \text{commission})
# \end{equation}
# 
# Where \( Q \) is the quantity traded.
# 

# In[ ]:


def apply_mean_reversion_strategy(df, stock,etf, window_size, initial_capital=1000, commission=0.000119):

 from sklearn.linear_model import LinearRegression
 win=30
 cap= float(1000)  #$1000
 commision=0.000119 #https://forum.alpaca.markets/t/pricing-and-fees/2309
 s_score=pd.Series()
 entry_price= []
 entry_time= []
 quantity=[]
 exit_time=[]
 exit_price=[]
 profits=[]
 rolling_cap= [cap]

 open_pos= False
 for i in range(1,len(df)-win):

     df_win= df.iloc[i-1:i+win].reset_index()
     ret= pd.DataFrame()
     ret[stock]= df_win[stock].pct_change()
     ret[etf]= df_win[etf].pct_change()
     timestamp= df_win['timestamp']
     ret.dropna(inplace=True)
     ret=ret.iloc[:win]

     tmp_ret = (ret - ret.mean())/ret.std()

     #Pg 45 Perform Linear Regression
     X = tmp_ret[etf]
     y = tmp_ret[stock] #stock return
     #print(X)
     #print(y)
     model1 = LinearRegression().fit(X.to_numpy().reshape(-1, 1),y)
     beta  = model1.coef_
     epsilon = y - model1.predict(X.to_numpy().reshape(-1, 1))

     #Define AR(1) process
     Xk = epsilon.cumsum() ## Cumulative sum of residuals to form the AR(1) process
     #second regression with it's lag
     X = Xk[:-1].to_numpy().reshape(-1,1)  # Lagged version of Xk (all but the last element)
     y = Xk[1:] # Current values of Xk (all but the first element)
     model2 = LinearRegression().fit(X,y)  # Fit a linear regression model to find AR(1) coefficients
     a = model2.intercept_ # Intercept of the AR(1) model
     b = model2.coef_ # Coefficient (slope) of the AR(1) model
     zeta = y - model2.predict(X)  # Residuals from the AR(1) model

     #OU parameters
     kappa = -np.log(b)*252
     kappa_rate = 1 if kappa > (252 / 10) else 0


     m = a/(1-b)
     sigma = np.sqrt(np.var(zeta)*2*kappa/(1-b**2))
     sigma_eq = np.sqrt(np.var(zeta)/(1-b**2))
     #print(-m/sigma_eq)


     s_score = pd.concat([s_score, pd.Series(-m / sigma_eq)])
     #print(f"{kappa_rate} of {s_score.iloc[-1]}")

     if(len(s_score)>1):
         upper_trading=single_level_crosses(s_score.tail(1),level=1.25)#level_crosses(s_score.tail(2), level=1.25)##
         lower_trading=single_level_crosses(s_score.tail(1),level=-1.25)#level_crosses(s_score.tail(2),level=-1.25)
         upper_stop_loss=level_crosses(s_score.tail(2), level=1.5)   #-1
         lower_stop_loss=level_crosses(s_score.tail(2), level=-1.5)#1
         upper_target= single_level_crosses(s_score.tail(1),level=0.5)#level_crosses(s_score.tail(2),level=0.5)
         lower_target= single_level_crosses(s_score.tail(1),level=-0.5)#level_crosses(s_score.tail(2),level=-0.75)

         if open_pos:
             if ((upper_stop_loss == 1 or upper_target ==1) and f==1) or ((lower_stop_loss == -1 or lower_target ==-1) and f==-1):
                 open_pos=False
                 trade_price=df_win[stock].iloc[win]
                 exit_price.append(trade_price)
                 #print(timestamp[win])
                 exit_time.append(timestamp[win])
                 #print(exit_time[-1])
                 if((upper_stop_loss ==1 or upper_target ==1) and f==1):
                     profits.append(entry_price[-1]-exit_price[-1])

                 elif((lower_stop_loss ==-1 or lower_target ==-1) and f==-1):
                     profits.append(exit_price[-1]-entry_price[-1])

                 rolling_cap.append(rolling_cap[-1] +(profits[-1]-commision)*quantity[-1])
                 #print(rolling_cap[-1])


         else:
             if(upper_trading==-1 or lower_trading==1) and kappa_rate==1:
                 open_pos=True
                 trade_price=df_win[stock].iloc[win]
                 entry_price.append(trade_price)
                 #print(trade_price)
                 entry_time.append(timestamp[win])
                 Qt= rolling_cap[-1]/trade_price
                 quantity.append(Qt)

                 if upper_trading == -1:
                     f=1
                 else:
                     f=-1
 return entry_price, entry_time, exit_time, exit_price, profits, rolling_cap


# In[ ]:


import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def apply_mean_reversion_strategy_monte_carlo(stock_df, etf_df, window_size, initial_capital=1000, commission=0.000119):
    def single_level_crosses(series, level):
        last = series.iloc[-1]
        if last > level:
            return 1
        elif last < -level:
            return -1
        return 0

    def level_crosses(series, level):
        if series.iloc[-2] <= level and series.iloc[-1] > level:
            return 1
        elif series.iloc[-2] >= level and series.iloc[-1] < level:
            return -1
        return 0

    entry_time = []
    entry_price = []
    exit_time = []
    exit_price = []
    profits = []
    rolling_cap = [initial_capital]  # Initialize rolling_cap with initial capital

    def process_simulation(stock_col, etf_col):
        open_pos = False
        quantity = []
        f = None

        for i in range(1, len(stock_df) - window_size):
            df_win = pd.DataFrame({
                'stock': stock_col.iloc[i-1:i+window_size].reset_index(drop=True),
                'etf': etf_col.iloc[i-1:i+window_size].reset_index(drop=True)
            })
            ret = df_win.pct_change().dropna().iloc[:window_size]

            if len(ret) < window_size:
                continue

            tmp_ret = (ret - ret.mean()) / ret.std()

            X = tmp_ret['etf'].to_numpy().reshape(-1, 1)
            y = tmp_ret['stock'].to_numpy()  # Convert to NumPy array

            model1 = LinearRegression().fit(X, y)
            beta = model1.coef_
            epsilon = y - model1.predict(X)

            Xk = epsilon.cumsum()  # No need to call .to_numpy()
            X = Xk[:-1].reshape(-1, 1)
            y = Xk[1:]

            model2 = LinearRegression().fit(X, y)
            a = model2.intercept_
            b = model2.coef_[0]  # Extract scalar coefficient

            zeta = y - model2.predict(X)

            kappa = -np.log(b) * 252
            kappa_rate = 1 if kappa > (252 / 10) else 0

            m = a / (1 - b)
            sigma = np.sqrt(np.var(zeta) * 2 * kappa / (1 - b ** 2))
            sigma_eq = np.sqrt(np.var(zeta) / (1 - b ** 2))

            s_score = pd.concat([pd.Series(-m / sigma_eq)], ignore_index=True)

            if len(s_score) > 1:
                upper_trading = single_level_crosses(s_score.tail(1), level=1.25)
                lower_trading = single_level_crosses(s_score.tail(1), level=-1.25)
                upper_stop_loss = level_crosses(s_score.tail(2), level=1.5)
                lower_stop_loss = level_crosses(s_score.tail(2), level=-1.5)
                upper_target = single_level_crosses(s_score.tail(1), level=0.5)
                lower_target = single_level_crosses(s_score.tail(1), level=-0.5)

                if open_pos:
                    if ((upper_stop_loss == 1 or upper_target == 1) and f == 1) or                             ((lower_stop_loss == -1 or lower_target == -1) and f == -1):
                        open_pos = False
                        trade_price = stock_col.iloc[i + window_size]
                        exit_price.append(trade_price)
                        exit_time.append(stock_df.index[i + window_size])

                        if (upper_stop_loss == 1 or upper_target == 1) and f == 1:
                            profits.append(entry_price[-1] - exit_price[-1])
                        elif (lower_stop_loss == -1 or lower_target == -1) and f == -1:
                            profits.append(exit_price[-1] - entry_price[-1])

                        rolling_cap.append(rolling_cap[-1] + (profits[-1] - commission) * quantity[-1])

                else:
                    if (upper_trading == -1 or lower_trading == 1) and kappa_rate == 1:
                        open_pos = True
                        trade_price = stock_col.iloc[i + window_size]
                        entry_price.append(trade_price)
                        entry_time.append(stock_df.index[i + window_size])

                        if rolling_cap:  # Ensure rolling_cap is not empty
                            quantity.append(rolling_cap[-1] / trade_price)
                        else:
                            quantity.append(0)

                        if upper_trading == -1:
                            f = 1
                        else:
                            f = -1

    # Apply the simulation for each column pair
    for stock_col, etf_col in zip(stock_df.columns, etf_df.columns):
        process_simulation(stock_df[stock_col], etf_df[etf_col])

    # Convert results to consistent format
    rolling_cap = pd.Series(rolling_cap).fillna(method='ffill').tolist()
    return entry_time, entry_price, exit_time, exit_price, profits, rolling_cap


# Mean reversion function to return the rolling capital only, for metrices calculations

# In[ ]:


def apply_mean_reversion_strategy_rc(df, stock,etf, window_size, initial_capital=1000, commission=0.000119):

  from sklearn.linear_model import LinearRegression
  win=30
  cap= float(1000)  #$1000
  commision=0.000119 #https://forum.alpaca.markets/t/pricing-and-fees/2309
  s_score=pd.Series()
  entry_price= []
  #entry_time= []
  quantity=[]
  #exit_time=[]
  exit_price=[]
  profits=[]
  rolling_cap = [float(cap)]

  open_pos= False
  for i in range(1,len(df)-win):

      df_win= df.iloc[i-1:i+win].reset_index()
      ret= pd.DataFrame()
      ret[stock]= df_win[stock].pct_change()
      ret[etf]= df_win[etf].pct_change()
      #timestamp= df_win['timestamp']
      ret.dropna(inplace=True)
      ret=ret.iloc[:win]

      tmp_ret = (ret - ret.mean())/ret.std()

      #Pg 45 Perform Linear Regression
      X = tmp_ret[etf]
      y = tmp_ret[stock] #stock return
      #print(X)
      #print(y)
      model1 = LinearRegression().fit(X.to_numpy().reshape(-1, 1),y)
      beta  = model1.coef_
      epsilon = y - model1.predict(X.to_numpy().reshape(-1, 1))

      #Define AR(1) process
      Xk = epsilon.cumsum()
      #second regression with it's lag
      X = Xk[:-1].to_numpy().reshape(-1,1)
      y = Xk[1:]
      model2 = LinearRegression().fit(X,y)
      a = model2.intercept_
      b = model2.coef_
      zeta = y - model2.predict(X)

        #OU parameters
      kappa = -np.log(b)*252
      kappa_rate = 1 if kappa > (252 / 10) else 0


      m = a/(1-b)
      sigma = np.sqrt(np.var(zeta)*2*kappa/(1-b**2))
      sigma_eq = np.sqrt(np.var(zeta)/(1-b**2))
      #print(-m/sigma_eq)


      s_score = pd.concat([s_score, pd.Series(-m / sigma_eq)])
      #print(f"{kappa_rate} of {s_score.iloc[-1]}")

      if(len(s_score)>1):
          upper_trading=single_level_crosses(s_score.tail(1),level=1.25)#level_crosses(s_score.tail(2), level=1.25)##
          lower_trading=single_level_crosses(s_score.tail(1),level=-1.25)#level_crosses(s_score.tail(2),level=-1.25)
          upper_stop_loss=level_crosses(s_score.tail(2), level=1.5)   #-1
          lower_stop_loss=level_crosses(s_score.tail(2), level=-1.5)#1
          upper_target= single_level_crosses(s_score.tail(1),level=0.5)#level_crosses(s_score.tail(2),level=0.5)
          lower_target= single_level_crosses(s_score.tail(1),level=-0.5)#level_crosses(s_score.tail(2),level=-0.75)

          if open_pos:
              if ((upper_stop_loss == 1 or upper_target ==1) and f==1) or ((lower_stop_loss == -1 or lower_target ==-1) and f==-1):
                  open_pos=False
                  trade_price=df_win[stock].iloc[win]
                  exit_price.append(trade_price)
                  #print(timestamp[win])
                  #exit_time.append(timestamp[win])
                  #print(exit_time[-1])
                  if((upper_stop_loss ==1 or upper_target ==1) and f==1):
                      profits.append(entry_price[-1]-exit_price[-1])

                  elif((lower_stop_loss ==-1 or lower_target ==-1) and f==-1):
                      profits.append(exit_price[-1]-entry_price[-1])

                  rolling_cap.append(rolling_cap[-1] +(profits[-1]-commision)*quantity[-1])
                  #print(rolling_cap[-1])


          else:
              if(upper_trading==-1 or lower_trading==1) and kappa_rate==1:
                  open_pos=True
                  trade_price=df_win[stock].iloc[win]
                  entry_price.append(trade_price)
                  #print(trade_price)
                  #entry_time.append(timestamp[win])
                  Qt= rolling_cap[-1]/trade_price
                  quantity.append(Qt)

                  if upper_trading == -1:
                      f=1
                  else:
                      f=-1
  return rolling_cap


# In[ ]:


def print_metrics_mean_reversion(df,etf):
  rolling_cap = apply_mean_reversion_strategy_rc(df, stock,etf, window_size=30)
  total_return, apr, sharpe, maxDD, maxDDD = calculate_metrics(np.array(rolling_cap))
  print(f"Total Return: {total_return * 100} %")
  print(f"APR: {apr * 100} %")
  print(f"Sharpe Ratio: {sharpe}")
  print(f"Max Drawdown: {maxDD}")


# In[ ]:


print_metrics_mean_reversion(df_15m,etf)


# In[ ]:


print_metrics_mean_reversion(df_30m,etf)


# In[ ]:


print_metrics_mean_reversion(df_1h,etf)


# Since for the Mean Reversion strategy, we are getting the best results for 15 minutes time frame, so we will proceed with that

# In[ ]:


entry_time, entry_price, exit_time, exit_price, profits, rolling_cap = apply_mean_reversion_strategy(df_15m,stock,etf,30)
df_pnl = trades_details(entry_time, entry_price, exit_time, exit_price, profits, rolling_cap)
plot(df_pnl,df_15m)


# Trade Details for the Mean Reversion Strategy

# In[ ]:


df_pnl


# #Both The Strategies Results on Monte Carlo Simulated Prices

# In[ ]:


import plotly.graph_objects as go

def plot_simulated_prices(simulated_prices,asset_name):
 num_simulations = 10
 days= len(simulated_prices[0])
 traces = []
 for i in range(num_simulations):
     traces.append(go.Scatter(
         x=np.arange(days),
         y=simulated_prices[i, :],
         mode='lines',
         name=f'Simulation {i+1}',
         line=dict(width=2),
         opacity=0.7
     ))


 fig = go.Figure(data=traces)

 fig.update_layout(
     title=f'Monte Carlo Simulation of {asset_name}',
     title_font=dict(size=20, color='black'),
     xaxis_title='Days',
     xaxis_title_font=dict(size=14, color='black'),
     yaxis_title='Price',
     yaxis_title_font=dict(size=14, color='black'),
     template='plotly_dark',
     legend_title='Simulations',
     legend=dict(
         title='Simulations',
         x=1.05,
         y=1,
         traceorder='normal',
         font=dict(size=12, color='white')
     ),
     margin=dict(l=0, r=150, t=40, b=0),
     paper_bgcolor='rgba(0,0,0,0)',
     plot_bgcolor='rgba(0,0,0,0)',
     xaxis=dict(
         showgrid=True,
         gridcolor='gray',
         gridwidth=0.5,
         tickfont=dict(color='white')
     ),
     yaxis=dict(
         showgrid=True,
         gridcolor='gray',
         gridwidth=0.5,
         tickfont=dict(color='white')
     )
 )

 fig.show()


# Monte Carlo Simulation is a statistical technique used to model and analyze complex systems or processes by generating a large number of random samples or simulations. It helps in understanding the behavior of a system under uncertainty and assessing the impact of risk or variability.
# 
# Monte Carlo Simulation involves the following steps:
# 
# Define a Model: Identify the system or process to be analyzed and define its mathematical model. This could include equations, parameters, and variables.
# 
# Generate Random Inputs: Use random sampling to generate a large number of possible values for the input variables based on their probability distributions.
# 
# Run Simulations: For each set of random inputs, run the model to compute the outcome. This process is repeated many times (thousands or millions of iterations) to create a distribution of possible outcomes.
# 
# Analyze Results: Aggregate and analyze the results of the simulations to gain insights into the behavior of the system, such as the distribution of possible outcomes, the likelihood of specific events, and the impact of uncertainty on the results.
# 
# Monte Carlo Simulation allows us to test how different scenarios (e.g., changes in volatility, mean return) affect the outcomes. This can be useful for stress testing and evaluating how robust your strategy or model is under various conditions.
# Optimization:
# 
# In our project, we have used Monte Carlo Simulation to:
# 
# Simulate Future Stock Prices: Generate multiple potential future paths for a stock’s price based on its historical volatility and mean return. This helps in predicting how the stock price might behave and assessing potential trading opportunities.
# 
# Evaluate Trading Strategies: By simulating various price paths, you can evaluate how well different trading strategies perform under different market conditions and uncertainty.
# 
# Assess Performance Metrics: Use simulated price paths to calculate performance metrics such as returns, risk, and drawdowns, providing insights into the effectiveness and robustness of your trading strategies.

# The monte_carlo_simulations_stock function simulates multiple potential future price paths for a given stock using the Monte Carlo method.
# 
# Initialize Parameters:
# 
# num_simulations: Number of simulated price paths to generate (set to 10).
# initial_price: The initial price of the asset, taken as the first value in the provided data frame df.
# days: Number of time steps in the simulation, corresponding to the length of the data frame.
# sigma: The annualized volatility of the asset, calculated as the standard deviation of the percentage change in prices, scaled by the square root of 252 (the number of trading days in a year).
# mu: The annualized mean return of the asset, calculated as the mean of the percentage change in prices, scaled by 252.
# Set Up Simulation:
# 
# simulated_prices: An array to store the simulated price paths, initialized with zeros. The first column is set to the initial price for all simulations.
# Run Simulations:
# 
# For each simulation (i), the process starts with the initial price.
# For each day (j), the function simulate_returns is used to generate the simulated return for that day.
# The price for the next day is computed as the current price multiplied by
# (
# 1
# +
# simulated return
# )
# (1+simulated return).
# Return Results:
# 
# The function returns the matrix simulated_prices, which contains all simulated price paths.

# 
# 
# Let \( S_0 \) be the initial price of the stock. We perform \( \text{num\_simulations} \) simulations over days (time frame). The volatility (standard-deviation) and the mean return are calculated as follows:
# 
# \begin{equation}
# \sigma = \text{std}\left(\frac{P_t - P_{t-1}}{P_{t-1}}\right) \times \sqrt{252}
# \end{equation}
# 
# \begin{equation}
# \mu = \text{mean}\left(\frac{P_t - P_{t-1}}{P_{t-1}}\right) \times 252
# \end{equation}
# 
# For each simulation \( i \) and day \( j \):
# 
# \begin{equation}
# \text{Simulated Return}_j = \text{simulate\_returns}(S_{i, j-1}, \mu, \sigma, \text{days})
# \end{equation}
# 
# \begin{equation}
# S_{i, j} = S_{i, j-1} \times \left(1 + \text{Simulated Return}_j\right)
# \end{equation}
# 
# The function returns a matrix of simulated prices containing all price paths.
# 
# 
# 

# In[ ]:


def monte_carlo_simulations_stock(df,asset_name):
  num_simulations = 10
  initial_price = df[asset_name].iloc[0]
  days = len(df)
  window_size = 30
  sigma= df[asset_name].pct_change().std() * np.sqrt(252)
  mu = df[asset_name].pct_change().mean() * 252

  simulated_prices = np.zeros((num_simulations, days))
  simulated_prices[:, 0] = initial_price

  for i in range(num_simulations):
    current_price = simulated_prices[i, 0]
    for j in range(1, days):
        simulated_return = simulate_returns(current_price, mu, sigma, days)
        simulated_prices[i, j] = simulated_prices[i, j - 1] * (1 + simulated_return.iloc[j-1])


  return simulated_prices


# Simulated Prices for 15 minute time frame, JP Morgan Stock Price for Mean Reversion

# In[ ]:


stock_simulated_prices = monte_carlo_simulations_stock(df_15m,'JPM')
stock_simulated_prices_df = pd.DataFrame(stock_simulated_prices)
stock_simulated_prices_df


# Simulated Prices for 15-minutes time frame for XLF ETF (Financial Select Sector SPDR Fund
# ) for Mean Reversion

# In[ ]:


etf_simulated_prices = monte_carlo_simulations_stock(df_15m,'XLF')
etf_simulated_prices_df = pd.DataFrame(etf_simulated_prices)
etf_simulated_prices_df


# Simulated Prices for 1 hour minute time frame, JP Morgan Stock Price for Trend Following

# In[ ]:


jpm_simulated_prices = monte_carlo_simulations_stock(df_1h,'JPM')
jpm_simulated_prices_df = pd.DataFrame(jpm_simulated_prices)
jpm_simulated_prices_df


# In[ ]:


plot_simulated_prices(jpm_simulated_prices,'JPM Stock Prices for Trend Following (1-hour time frame)')


# In[ ]:


plot_simulated_prices(stock_simulated_prices,'JPM Stock Price for Mean Reversion (15-minute time frame)')


# In[ ]:


plot_simulated_prices(etf_simulated_prices,'XLF Price for Mean Reversion (15-minute time frame)')


# ## Trend Following Strategy

# Getting the details of Rolling Capital after executing Trades on Monte Carlo Simulated Price of JP Morgan (1-hour time frame)

# In[ ]:


trend_following_rolling_capital = []
num_simulations = 10
window_size = 30
short_window = 12
long_window = 26
signal_window = 9
rsi_window = 14
ema_window = 20
for i in range(num_simulations):
    df_simulated = pd.DataFrame({'JPM': stock_simulated_prices[i]})
    df_simulated = calculate_macd(df_simulated,short_window,long_window, signal_window)
    df_simulated = calculate_rsi(df_simulated,rsi_window)
    df_simulated = calculate_ema(df_simulated,ema_window)
    result_trend = apply_trend_following_strategy_rc(df_simulated, 'JPM', window_size,short_window,long_window,signal_window,rsi_window,ema_window, initial_capital=1000, commission=0.000119)
    trend_following_rolling_capital.append(result_trend)

print(trend_following_rolling_capital)


# In[ ]:


jpm_simulated_prices_reshaped = jpm_simulated_prices.reshape(10, -1)
jpm_simulated_prices_reshaped_df = pd.DataFrame(jpm_simulated_prices_reshaped)
jpm_simulated_prices_reshaped_df= jpm_simulated_prices_reshaped_df.T
jpm_simulated_prices_reshaped_df


# In[ ]:


# Assuming apply_trend_following_strategy_monte_carlo has been called
results = apply_trend_following_strategy_monte_carlo(jpm_simulated_prices_reshaped_df, 30, short_window, long_window, signal_window, rsi_window, ema_window)

# Initialize a dictionary to store the PnL DataFrames for each stock
df_pnls = {}

# Loop over the results dictionary
for stock, stock_results in results.items():
    # Extract results for the current stock
    entry_time = stock_results['entry_time']
    entry_price = stock_results['entry_price']
    exit_time = stock_results['exit_time']
    exit_price = stock_results['exit_price']
    profits = stock_results['profits']
    rolling_cap = stock_results['rolling_cap']

    # Generate the trades PnL DataFrame for this stock
    df_pnl = trades_details(entry_time, entry_price, exit_time, exit_price, profits, rolling_cap)

    # Store the resulting PnL DataFrame in the dictionary
    df_pnls[stock] = df_pnl
    df_pnls[stock]


# In[ ]:


results


# In[ ]:


# Iterate over Rolling Capital and calculate metrics
for i in range(len(df_pnl['Rolling Capital'])):
    subset_rolling_capital = np.array(df_pnl['Rolling Capital'][:i+1])  # Subset including up to the current index
    total_return, apr, sharpe, max_dd, _ = calculate_metrics(subset_rolling_capital)

    print(f"Metrics for Rolling Capital up to index {i}:")
    print(f"Total Return: {total_return * 100:.2f} %")
    print(f"APR: {apr * 100:.2f} %")
    print(f"Sharpe Ratio: {sharpe:.2f}")
    print(f"Max Drawdown: {max_dd:.2f}")
    print("----")


# In[ ]:


#Mean metrices of all the 10 Monte Carlo Simulations

# Calculate metrics for each simulation and store them in a list
all_metrics = []
for rolling_cap in trend_following_rolling_capital:
    total_return, apr, sharpe, maxDD, _ = calculate_metrics(np.array(rolling_cap))
    all_metrics.append([total_return, apr, sharpe, maxDD])

# Convert the list of metrics to a NumPy array for easy calculations
all_metrics_np = np.array(all_metrics)

# Calculate the mean of each metric across all simulations
mean_total_return = np.mean(all_metrics_np[:, 0])
mean_apr = np.mean(all_metrics_np[:, 1])
mean_sharpe = np.mean(all_metrics_np[:, 2])
mean_maxDD = np.mean(all_metrics_np[:, 3])

# Print the mean metrics
print("Mean Metrics for 10 Monte Carlo Simulations:")
print(f"Mean Total Return: {mean_total_return * 100:.2f} %")
print(f"Mean APR: {mean_apr * 100:.2f} %")
print(f"Mean Sharpe Ratio: {mean_sharpe:.2f}")
print(f"Mean Max Drawdown: {mean_maxDD:.2f}")


# Rolling Capital Percentage Change Plot for Trend Following Strategy

# In[ ]:


trend_following_rolling_capital_series = pd.DataFrame(trend_following_rolling_capital).T
ax = sns.kdeplot(data=trend_following_rolling_capital_series.pct_change().iloc[1:])
plt.show()


# ##Mean Reversion Strategy

# Getting the details of Rolling Capital after executing Trades on Monte Carlo Simulated Prices of JP Morgan (15-minutes time frame) and XLF Prices (15-minutes time frame)

# In[ ]:


mean_reversion_rolling_capital = []
for i in range(num_simulations):
    # Ensure both arrays have the same length before creating the DataFrame
    min_length = min(len(jpm_simulated_prices[i]), len(etf_simulated_prices[i]))
    df_simulated_mr = pd.DataFrame({
        'JPM': jpm_simulated_prices[i][:min_length],
        'XLF': etf_simulated_prices[i][:min_length]
    })
    result_mean = apply_mean_reversion_strategy_rc(df_simulated_mr, 'JPM', 'XLF', window_size, initial_capital=1000, commission=0.000119)
    mean_reversion_rolling_capital.append(result_mean)

print(mean_reversion_rolling_capital)


# In[ ]:


stock_simulated_prices_reshaped = stock_simulated_prices.reshape(10, -1)
stock_simulated_prices_reshaped_df = pd.DataFrame(stock_simulated_prices_reshaped.T)
stock_simulated_prices_reshaped_df


# In[ ]:


etf_simulated_prices_reshaped = etf_simulated_prices.reshape(10, -1)
etf_simulated_prices_reshaped_df = pd.DataFrame(etf_simulated_prices_reshaped.T)
etf_simulated_prices_reshaped_df


# In[1]:


#entry_time, entry_price, exit_time, exit_price, profits, rolling_cap = apply_mean_reversion_strategy_monte_carlo(stock_simulated_prices_reshaped_df, etf_simulated_prices_reshaped_df, window_size)
#df_pnl = trades_details(entry_time, entry_price, exit_time, exit_price, profits, rolling_cap)
#df_pnl


# In[2]:


#total_return, apr, sharpe, maxDD, maxDDD = calculate_metrics(np.array(mean_reversion_rolling_capital))
#print(f"Total Return: {total_return * 100} %")
#print(f"APR: {apr * 100} %")
#print(f"Sharpe Ratio: {sharpe}")
#print(f"Max Drawdown: {maxDD}")


# Rolling Capital Percentage Change Plot for Mean Reversion Strategy

# In[3]:


#mean_reversion_rolling_capital_series = pd.DataFrame(mean_reversion_rolling_capital).T
#ax = sns.kdeplot(data=mean_reversion_rolling_capital_series.pct_change().iloc[1:])
#plt.show()

