# Live Trading: Strategy Execution  
from datetime import datetime, date, timedelta
import pandas as pd
import time
 

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import  MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

import warnings
warnings.filterwarnings("ignore")

 

# Get the tokens for API access
api_key = 'your_api_key'  
secret_key = 'your_secret_key' 

# instatiate the trading client
api = TradingClient(api_key, secret_key, raw_data=True) # raw_data = True returns python dict and not convert the output into object

# close any open positions 
# api.close_all_positions(cancel_orders=True)

# clock
clock = api.get_clock()
# get the account details
account = api.get_account()
# get a list of all of our positions.
portfolio = api.get_all_positions()

current_timestamp = datetime.now()
end_date = current_timestamp - timedelta(hours=6)
start_date=end_date- timedelta(days=30)

stock='JPM'
etf='XLF' 
cap = float(1000)  # $1000 
results=pd.DataFrame()



# Trading App
class App(): 
    
    def __init__(self,cap): 
        self.cap=cap
        self.quantity = []
        self.entry_price = []
        self.entry_time = []
        self.exit_time = []
        self.exit_price = []
        self.profits = []
        self.rolling_cap = [cap]
        self.commission = 0.000119
        self.open_pos = False
        self.position=0
        self.win = 30
        print(self.open_pos)

    
    def calculate_macd(self,df, short_window=69, long_window=48, signal_window=55):
        df['MACD'] = df[stock].ewm(span=short_window, adjust=False).mean() - df[stock].ewm(span=long_window, adjust=False).mean()
        df['MACD_signal'] = df['MACD'].ewm(span=signal_window, adjust=False).mean()
        df['MACD_hist'] = df['MACD'] - df['MACD_signal']
        return df

# Calculate RSI
    def calculate_rsi(self,df, window=81):
        delta = df[stock].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        return df

    # Calculate EMA
    def calculate_ema(self,df, window=19):
        df['EMA'] = df[stock].ewm(span=window, adjust=False).mean()
        return df 

    def transform_data(self):
        stock_client = StockHistoricalDataClient(api_key, secret_key)
        data = stock_client.get_stock_bars(
                StockBarsRequest(
                symbol_or_symbols = [stock, etf],
                start = start_date,
                end = end_date,
                timeframe = TimeFrame(1, TimeFrameUnit.Hour))).df
        data.reset_index(inplace=True) 
        stock_df = pd.DataFrame()
        etf_df = pd.DataFrame()
        stock_df[stock] = data[data['symbol'] == stock].set_index('timestamp')['close']
        etf_df[etf] = data[data['symbol'] == etf].set_index('timestamp')['close']

        df = pd.merge(etf_df[[etf]], stock_df[[stock]], left_index=True, right_index=True, how='outer')
        df.fillna(method='bfill', inplace=True)
        return df
    

    def run_trend(self):


        if clock.get('is_open') == True:
            # get data
            df_trend= self.transform_data()
             
            df_trend = self.calculate_macd(df_trend)
            df_trend = self.calculate_rsi(df_trend)
            df_trend = self.calculate_ema(df_trend)
            
            macd_signal = df_trend['MACD'].iloc[-1]
            macd_hist = df_trend['MACD_hist'].iloc[-1]
            rsi = df_trend['RSI'].iloc[-1]
            price = df_trend[stock].iloc[-1]
            #sma = df_win['SMA'].iloc[-1]
            sma = df_trend['EMA'].iloc[-1]
            print(macd_hist,price,sma,rsi)
    # Trend-following strategy
        #portfolio = api.get_all_positions()
            if not self.open_pos:
                if macd_hist > 0 and df_trend[stock].iloc[-1] < sma and rsi < 70:
                    self.open_pos = True
                    self.position = 1  # Long position 
                    
                    api.submit_order(MarketOrderRequest(symbol = stock, qty = int(self.rolling_cap[-1] / price), side = OrderSide.BUY, time_in_force = TimeInForce.DAY))
                    time.sleep(1) 
                    exec_order=api.get_orders()[0]
                    print(f"Going long with average price of {exec_order['filled_avg_price']}")
                    self.entry_price.append(exec_order['filled_avg_price'])
                    self.quantity.append(exec_order['filled_qty'])
                    self.entry_time.append(exec_order['submitted_at']) 
                    

                elif macd_hist < 0 and df_trend[stock].iloc[-1] > sma and rsi > 30: 
                    self.open_pos = True
                    self.position = -1  # Short position 
                     
                    api.submit_order(MarketOrderRequest(symbol = stock, qty = int(self.rolling_cap[-1] / price), side = OrderSide.SELL, time_in_force = TimeInForce.DAY))
                    time.sleep(1)  
                    exec_order=api.get_orders()[0] 
                    self.entry_price.append(exec_order['filled_avg_price'])
                    print(f"Going Short with average price of {exec_order['filled_avg_price']}")
                    self.quantity.append(exec_order['filled_qty'])
                    self.entry_time.append(exec_order['submitted_at']) 
                    

            if self.open_pos:
                if (self.position == 1 and (macd_hist < 0 or rsi > 70)) or (self.position == -1 and (macd_hist > 0 or rsi < 30)):
                    self.open_pos = False
                    self.exit_price.append(price)
                    self.exit_time.append(df_trend.index[-1])
                    if self.position == 1:
                        api.close_all_positions(cancel_orders=True)
                        balance_change=float(account.get('equity')) - float(account.get('last_equity'))
                        self.profits.append(balance_change)
                    else:
                        api.close_all_positions(cancel_orders=True)
                        balance_change=float(account.get('equity')) - float(account.get('last_equity'))
                        self.profits.append(balance_change)
 
                    exec_order=api.get_orders()[0]
                    print(exec_order['filled_avg_price'])
                    self.exit_price.append(exec_order['filled_avg_price'])
                    print(self.exit_price)
                    self.quantity.append(exec_order['filled_qty'])
                    self.exit_time.append(exec_order['submitted_at']) 
                    self.rolling_cap.append(self.rolling_cap[-1]+self.profits[-1])
                    self.position = 0

                    if(len(self.profits)>100): 
                        min_length = min(len(self.entry_time), len(self.entry_price), len(self.exit_time),
                                len(self.exit_price), len(self.profits), len(self.rolling_cap[1:]))

                        df_pnl = pd.DataFrame({
                        "Entry_time": self.entry_time[:min_length],
                        "Entry_price": self.entry_price[:min_length],
                        "Exit_time": self.exit_time[:min_length],
                        "Exit_price": self.exit_price[:min_length],
                        'pnl': self.profits[:min_length],
                        'Rolling Capital': self.rolling_cap[1:min_length+1]  # +1 to keep the slice consistent
                        })
                        print("Logging trade details")
                        df_pnl.to_csv("trading_sheet.csv")
                # signal logic
                
            
        else: 
            balance_change=float(account.get('equity')) - float(account.get('last_equity'))
            #balance_change = float(account.equity) - float(account.last_equity)
            print("Market Closed")
            print(f"Today's portfolio balance change is : {balance_change}")

                

 
if __name__ == "__main__":  
    app = App(cap)
    while clock.get('is_open'):
        end_date = current_timestamp - timedelta(hours=6)
        start_date=end_date- timedelta(days=30)
        app.run_trend()
        time.sleep(10) 
    else:
        print(f"Market is closed, Next opens at {clock.get('next_open')}")
        print(account.get())
        balance_change = float(account.get('equity')) - float(account.get('last_equity'))
        print(f"Today's portfolio balance change is : {balance_change:.2f}")      
