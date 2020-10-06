import stock_mgt as sm
import ticker_mgt as tm
import optimal_mgt as om
import LSTM_mgt as lm
if True:
    sm.drop_stock_table()
    sm.create_stock_table()
    sm.download_stock()
    tm.drop_ticker_table()
    tm.create_ticker_table()
    tm.download_ticker()
    om.plan_optimization()
    lm.drop_forecast_table()
    lm.create_forecast_table()
    lm.prediction()