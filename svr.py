import pandas as pd
import numpy as np
import math
from sklearn.svm import SVR
import matplotlib.pyplot as plt

# 算mean sqrt比較svr跟lstm

def get_data(filename): 
    dates = []
    prices = []
    

    data = pd.read_csv(filename)
    train = data['open']
    train= train.values.reshape(-1,1)
    for i in range(10,len(train)):
        dates.append(train[i-10:i-1,0])
        prices.append(train[i,0])

    dates = np.array(dates)
    prices = np.array(prices)
    #dates = np.reshape(dates, (dates.shape[0], dates.shape[1], 1))
    return dates,prices,data['open']
 
def predict_price(dates, prices,file):  
    #dates = np.reshape(dates, (len(dates), 1)) 
    svr_rbf = SVR(kernel = 'rbf', C = 1e3, gamma = 0.1)
    svr_rbf.fit(dates, prices)

    
    #plt.scatter(dates, prices, color = 'black', label = 'Data')
    test = file[20:]
    test = test.values.reshape(-1,1)
    

    # 預測四月
    april = pd.read_csv(r'D:\股票csv\2603_4.csv')
    
    tmp = pd.DataFrame(test)
    temp = pd.concat((tmp, april['open']),axis=0)
    temp = np.array(temp)     #四月的實際股價
  
    for i in range(20):
        X = []
        for j in range(10, len(test)):
            X.append(test[j-10:j-1,0])
        X = np.array(X)
        #Y = np.reshape(X, (X.shape[0], X.shape[1], 1))
        predict = svr_rbf.predict(X)
        predict = np.array(predict)
        predict.reshape(-1,1)
        
        tmp_io = pd.DataFrame(test)
        tmp_pd = pd.DataFrame(predict)
        tmp_io = pd.concat((tmp_io,tmp_pd.iloc[len(tmp_pd)-1]),axis=0)
        #print(X)
        test= tmp_io.values
        test = np.array(test)
        test.reshape(-1,1)
        #print('pre',pre)
        print('a')

    a = predict[len(predict)-19:]
    plt.plot(april['open'], color = 'red', label = 'Real')
    plt.plot(a, color = 'blue', label = 'SVR Model')
    
    plt.xlabel('Date')
    plt.ylabel("Price")
    plt.title("Support Vector Regression (SVR)")
    plt.legend()
    plt.show()
    
    
    diff = math.sqrt(np.mean((test-predict)**2))
    
    print('Diff :',diff)

    return predict[len(predict)-11]

X,Y,data = get_data(r"D:\股票csv\2603.csv")

#predict_price(X,Y)
predicted_price = predict_price(X,Y,data)
print("The Stock Open Price: ")
print("RBF Kernel: $", (predicted_price))