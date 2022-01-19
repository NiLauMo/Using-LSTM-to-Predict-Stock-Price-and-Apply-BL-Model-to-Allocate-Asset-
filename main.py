import numpy as np
import pandas as pd
import os
import seaborn as sns
import datetime
from pypfopt import black_litterman
from pypfopt import risk_models as RiskModels
from pypfopt.black_litterman import BlackLittermanModel
from sklearn.preprocessing import MinMaxScaler  
import keras
from keras.models import Sequential
from keras.layers.core import *
from keras.layers import Dense, LSTM, Activation
from keras import optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
from autograd import jacobian, hessian
from sklearn.utils import shuffle

class LMODEL:
    def __init__(self, filename, index):
        self.index = index
        self.filename = filename
        test = filename[filename.date>'2020/6/10']
        train = filename[:len(filename)-len(test)]
        # label
        train_set = train['open']

        # normolize
        sc = MinMaxScaler(feature_range = (0, 1))
        train_set= train_set.values.reshape(-1,1)
        train_set = sc.fit_transform(train_set)

        X_train = [] 
        Y_train = []
        for i in range(20,len(train_set)):
            X_train.append(train_set[i-20:i-1, 0]) # 10個1組
            Y_train.append(train_set[i, 0])
        X_train, Y_train = np.array(X_train), np.array(Y_train) 
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))  # 2 dimension to 3 dim

        keras.backend.clear_session()
        regressor = Sequential()
        # print('x_train',X_train.shape[1])
        
        regressor.add(LSTM(units = 100, input_shape = (X_train.shape[1], 1)))
        
        regressor.add(Dense(units = 1))
        regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

        callback = EarlyStopping(monitor = "loss", patience=10, verbose=1, mode="auto")
        history = regressor.fit(X_train, Y_train, epochs = 30, batch_size = 16, callbacks= callback) #先調3比較快
        
        dataset = pd.concat((train['open'],test['open']),axis = 0)
        

        inputs = dataset[len(dataset) - len(test) - 20:].values
        inputs = inputs.reshape(-1,1)   # len(input)*1

        # print('inputs:',inputs)
        inputs = sc.transform(inputs)
        max = 0

        for a in range(19):
            X_test = []
            for i in range(20, len(inputs)):
              X_test.append(inputs[i-20:i-1, 0])
            X_test = np.array(X_test)
            Y_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        
            # predict
            predicted_data = regressor.predict(Y_test)
            num = len(predicted_data)-1
            if predicted_data[num]>max:
               max = predicted_data[num]   
            tmp_io = pd.DataFrame(inputs)
            tmp_pd = pd.DataFrame(predicted_data)
            tmp_io = pd.concat((tmp_io,tmp_pd.iloc[len(tmp_pd)-1]),axis=0)
            inputs = tmp_io.values
            inputs = np.array(inputs)
            inputs = inputs.reshape(-1,1)  
            #inputs.append('1')  # 暫停用

        

        max = np.array(max)
        max = max.reshape(-1,1)
        max = sc.inverse_transform(max)
        
        predicted_data = sc.inverse_transform(predicted_data)
        self.predict = predicted_data
        self.pre_max = max[0]
        print('stock: ',index, ' max= ',max)

        April = pd.read_csv(r'2603_4.csv')
        tmp = predicted_data[len(predicted_data)-20:]
        plt.plot(test['open'].values, color = 'black', label = 'Real Stock Price')
        #plt.plot(tmp, color = 'green', label = 'Predicted Stock Price') #只要看四月的話
        plt.plot(predicted_data, color = 'green', label = 'Predicted Stock Price')
        plt.title('Stock Price Prediction')
        plt.xlabel('Time')
        plt.ylabel('Stock Price')
        plt.legend()

    def get_pred_price(self):
        last = np.array(self.filename['open'])
        last = last[len(last)-1]
        val = self.pre_max-last
        val = val / last

        print('stock: ',self.index,' rate of return= ',val)
        return val



    ###------------ required return ------------###
def get_omega(self, tau, P, sigma):
        omega = tau * P @ sigma @ P.T
        assert omega.shape[0] == self.K, "[Error]: Row dimension of omega should be {0}, but {1}".format(self.K, omega.shape[0])
        assert omega.shape[1] == self.K, "[Error]: Col dimension of omega should be {0}, but {1}".format(self.K, omega.shape[1])
        return omega

#New version
class BL_Model:
    def __init__(self, P, Q):
        prices = pd.read_csv(r'total.csv')
        tse = pd.read_csv(r"指數.csv")

        prices['Date'] = pd.to_datetime(prices['Date'])
        prices = prices.set_index('Date')
        #self.prices.drop(0)
        currencies = ['2002', '2330', '2603', '2881']

        for currency in currencies:
            prices[currency] = prices[currency].astype(float)

        #print(prices)
        shrunk_covariance = RiskModels.CovarianceShrinkage(prices)        # a class
        shrunk_covariance = shrunk_covariance.shrunk_covariance()    # np.array
        
        weight_set = []
        shrunk_covariance = np.array(shrunk_covariance)
        print("P")
        print(P)
        print('Q',Q)
        print('cov')
        print(shrunk_covariance)

        delta = black_litterman.market_implied_risk_aversion(tse["TSE"].astype(float))
        Omega = np.dot(np.dot(np.dot(0.05, np.linalg.inv(P)),shrunk_covariance),P)
        print(Omega)
        #Omega = BlackLittermanModel.default_omega(cov_matrix = shrunk_covariance, P = P, tau = 0.05)
        bl = BlackLittermanModel(shrunk_covariance, P = P, Q = Q, omega = Omega)
        ret = bl.bl_returns()
        bl.bl_weights(delta)
        self.weight = bl.clean_weights()
        d = []
        print('Weight')
        print(self.weight)

        for i,j in self.weight.items():
            np.insert(d,0,self.weight[i])
        print('D')
        print(d)

        x = np.arange(len(currencies))
        plt.bar(x, [self.weight[0],self.weight[1],self.weight[2],self.weight[3]], color=['red','green','blue','yellow'])
        plt.xticks(x, currencies)
        plt.xlabel('Stock')
        plt.ylabel('Weight')
        plt.title('BL Model asset allocation')
        plt.show()
        #plt.close()
        
    def get_weight(self):
        return self.weight
    
    #def get_plot(self):
    #    self.df.T.plot(kind='bar') # 原本是bar
    #    plt.show()


def get_lstm():
    stock3 = pd.read_csv(r"2603.csv")
    stock1 = pd.read_csv(r"2002.csv")
    stock4 = pd.read_csv(r"2881.csv")
    stock2 = pd.read_csv(r"2330.csv")

    one_model = LMODEL(stock1,2002)
    two_model = LMODEL(stock2,2330)
    three_model = LMODEL(stock3,2603)
    four_model = LMODEL(stock4,2881)

    P = [
        [1,0,0,0],
        [0,1,0,0],
        [0,0,1,0],
        [0,0,0,1]
        ]

    Q = np.array([float(one_model.get_pred_price()), float(two_model.get_pred_price()), float(three_model.get_pred_price()), float(four_model.get_pred_price())])

    print(P)
    print(Q)
    return P,Q

def get_report():
    s2002 = float((31 - 25.9)/25.9)
    s2881 = float((63 - 56.8)/56.8)
    s2603 = float((58 - 45.5)/45.5)
    s2330 = float((650 - 587)/587)

    P = [
        [1,0,0,0],
        [0,1,0,0],
        [0,0,1,0],
        [0,0,0,1]
        ]
    Q = np.array([s2002, s2330, s2603, s2881])


    return P,Q

def get_cc_return():
    df = pd.read_csv(r'total.csv')
    log_ret = np.log(df/df.shift()) # shift(): Move one unit downward
    log_ret = log_ret.drop(index=[0])
    log_ret = np.array(log_ret)
    ret =[
        [log_ret[3542][0],0,0,0],
        [0,log_ret[3542][1],0,0],
        [0,0,log_ret[3542][2],0],
        [0,0,0,log_ret[3542][3]]
        ]

    return ret

def report_bl():
    P,Q = get_report()
    BL2 = BL_Model(P,Q)
    w = BL2.get_weight()


def lstm_bl():
    P,Q = get_lstm()
    #get_lstm()
    plt.show()
    #plt.close()
    BL = BL_Model(P,Q)
    w = BL.get_weight()

lstm_bl()
os.system('pause')
report_bl()
