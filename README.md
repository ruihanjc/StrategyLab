# StrategyLab
A naive systematic trading approach from data-pipeline to trading rules


strategy_data :
- A tool I use to store market data from different sources, then store inside the ArcticDB which is the databse I use to store my data.
- To initialize the sys_data, first initialize the ArcticDB using arctic_init.py.
- You can also delete the Arcticdb using arctic_delete.py.
- Using the standard_update_data.py and changes in the config.yaml file, you can configure strategy_data to fetch the data you want.


strategy_viz:
- A tool used to have a visualization of the current data we have, it uses streamlit, so to initialize, run the following code:
            `streamlit run strategy_viz/app.py`


strategy_backtest:
- Backtesting tool to check the return from a certain rule, the rule resides in strategy_core If you want to add one on your own.
- Run the main.py file to initialize a backtest.


strategy_brokers:
- Package used to create clients that I can use to trade.

strategy_production:
- Paper trading engine that uses IBKR's api and strategy_core's classes to identify signals, buy or sell when signal notified.

- The normal workflow works like the following:
    
    - Standard update data
    - Account update
    - Daily signal
    - Check signals and order
    - Execution


Additional materials:

- I personally have a couple of cron tasks to use that updates data everyday, and some other small tasks so it's easier.
- The system is ran in another PC and it's purpose is solely for trading.
