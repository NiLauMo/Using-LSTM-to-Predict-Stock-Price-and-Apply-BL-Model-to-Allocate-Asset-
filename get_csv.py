from yahoo_fin.stock_info import *

stock_no = '5222.TW'

data = get_data(stock_no , start_date = '2021/11/01', end_date = '2021/11/30')
data.head()
data.to_csv('5222_4.csv')

