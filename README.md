# Using-LSTM-to-Predict-Stock-Price-and-Apply-BL-Model-to-Allocate-Asset-

Before Run this Code, you should check python including the following header, if not, please use 'python -m pip install' to get them.
1. numpy
2. pandas
3. seaborn
4. PyPortfolioOpt
5. keras
6. autograd
7. scikit-learn

After you check these component are avaliable, you can run this code.

Brief Description
In stock market, it is important to spread the risk. Therefore, I apply Black-Litterman Model in my work. Black-Litterman Model(BL Model) is one of the best way of Asset Allocation. By computing the rate of return of each stock as input, then we can get the configuration of asset allocation. According to the estimate of rate of return, this is the best combination. However, what is the method to get the estimate of rate of return? There are two methods. First, I used LSTM model to predict. Using the history stock price from 2018/3 to 2021/3 to train model, then predicting the whole month on April, 2021. Second, I used the target price from investment company/bank

See the whole document, please go to the file "Using LSTM to Predict Stock Price and Apply Black-Litterman Model to Do Asset Model"
