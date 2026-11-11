# Live Trading: Strategy Execution  
 
from datetime import datetime, date, timedelta
import pandas as pd
import numpy as np
import time
import warnings
warnings.filterwarnings("ignore")

 

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import  MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

from sklearn.linear_model import LinearRegression

 

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
cap=1000
stock='JPM'
etf='XLF' 

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
        self.s_score=pd.Series()
        print(self.open_pos)
        print(self.open_pos) 

# Calculate RSI
    def calculate_rsi(self,df, window=81):
        delta = df[stock].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        return df

    def transform_data(self):
        stock_client = StockHistoricalDataClient(api_key, secret_key)
        data = stock_client.get_stock_bars(
                StockBarsRequest(
                symbol_or_symbols = [stock, etf],
                start = start_date,
                end = end_date,
                timeframe = TimeFrame(15, TimeFrameUnit.Minute))).df
        data.reset_index(inplace=True) 
        stock_df = pd.DataFrame()
        etf_df = pd.DataFrame()
        stock_df[stock] = data[data['symbol'] == stock].set_index('timestamp')['close']
        etf_df[etf] = data[data['symbol'] == etf].set_index('timestamp')['close']

        df = pd.merge(etf_df[[etf]], stock_df[[stock]], left_index=True, right_index=True, how='outer')
        df.fillna(method='bfill', inplace=True)
        return df
    
    def level_crosses(self,series, level):  #double check
        series=series.values
        change = 0
        if series[1]>level and series[0]<level:
            change=1
        elif series[1]<level and series[0]>level:
            change=-1
        else:
            change=0
        return change
    
    def single_level_crosses(self,series, level):  #single check
        series=series.values
        change = 0
        if series[0]>level:
            change=-1
        elif series[0]<level:
            change=1
        else:
            change=0
        return change

    def run_mean_reversion(self):


        if clock.get('is_open') == True:
            # get data
            s_score=self.s_score
            df= self.transform_data() 
            df= self.calculate_rsi(df)  
            rsi = df['RSI'].iloc[-1]
            price = df[stock].iloc[-1] 

            df_win= df.iloc[-self.win:].reset_index()
            ret= pd.DataFrame()
            ret[stock]= df_win[stock].pct_change()
            ret[etf]= df_win[etf].pct_change()
            timestamp= df_win['timestamp'] 
            ret.dropna(inplace=True)
            ret=ret.iloc[:self.win]

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


            self.s_score = pd.concat([self.s_score, pd.Series(-m / sigma_eq)])

            if(len(self.s_score)>1):
                upper_trading=self.single_level_crosses(self.s_score.tail(1),level=1.25)#level_crosses(s_score.tail(2), level=1.25)##
                lower_trading=self.single_level_crosses(self.s_score.tail(1),level=-1.25)#level_crosses(s_score.tail(2),level=-1.25)
                upper_stop_loss=self.level_crosses(self.s_score.tail(2), level=1.5)
                lower_stop_loss=self.level_crosses(self.s_score.tail(2), level=-1.5)
                upper_target= self.single_level_crosses(self.s_score.tail(1),level=0.5)#level_crosses(s_score.tail(2),level=0.5)
                lower_target= self.single_level_crosses(self.s_score.tail(1),level=-0.5)#level_crosses(s_score.tail(2),level=-0.75)
            #print(f"{kappa_rate} of {s_score.iloc[-1]}")
                if self.open_pos:
                    if ((upper_stop_loss == 1 or upper_target ==1) and f==1) or ((lower_stop_loss == -1 or lower_target ==-1) and f==-1):
                        self.open_pos=False
                        
                        api.close_all_positions(cancel_orders=True) 
                        
                        balance_change=float(account.get('equity')) - float(account.get('last_equity'))
                        self.profits.append(balance_change)

                        exec_order=api.get_orders()[0]
                        print(exec_order['filled_avg_price'])
                        self.exit_price.append(exec_order['filled_avg_price'])
                        print(f"Exiting at{self.exit_price[-1]}")
                        self.quantity.append(exec_order['filled_qty'])
                        self.exit_time.append(exec_order['submitted_at'])
                        
                        self.rolling_cap.append(self.rolling_cap[-1] + balance_change)

                        if(len(self.profits)>1000):
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
                            df_pnl.to_csv("trading_sheet_mean.csv")
                    
                        #print(rolling_cap[-1])
                            
                
                else:
                    if(upper_trading==-1 or lower_trading==1) and kappa_rate==1:
                        self.open_pos=True
                        price=df_win[stock].iloc[-1]
                        
                        if upper_trading == -1:
                            api.submit_order(MarketOrderRequest(symbol = stock, qty = int(self.rolling_cap[-1] / price), side = OrderSide.SELL, time_in_force = TimeInForce.DAY))
                            f=1
                            print("Buy")
                        else:
                            api.submit_order(MarketOrderRequest(symbol = stock, qty = int(self.rolling_cap[-1] / price), side = OrderSide.BUY, time_in_force = TimeInForce.DAY))
                            f=-1
                            print("Sell")
                        exec_order=api.get_orders()[0]
                        #print(exec_order['filled_avg_price'])
                        self.entry_price.append(exec_order['filled_avg_price'])
                        print(self.entry_price)
                        self.quantity.append(exec_order['filled_qty'])
                        self.entry_time.append(exec_order['submitted_at'])
    
                
            
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
        app.run_mean_reversion()
        #time.sleep(300) 
    else:
        print(f"Market is closed, Next opens at {clock.get('next_open')}")
        print(account.get())
        balance_change = float(account.get('equity')) - float(account.get('last_equity'))
        print(f"Today's portfolio balance change is : {balance_change:.2f}")      
