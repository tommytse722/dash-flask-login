import stock_mgt as sm
import ticker_mgt as tm
if True:
    sm.drop_stock_table()
    sm.create_stock_table()
    sm.download_stock()
    tm.drop_ticker_table()
    tm.create_ticker_table()
    tm.download_ticker()