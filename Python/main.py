from pandas_datareader import data as pdr
import numpy as np
import yfinance as yf
import datetime
import matplotlib.pyplot as plt
import matplotlib.cm as cm

yf.pdr_override()

startDate = datetime.datetime(2023, 1, 1)
endDate = datetime.datetime(2023, 2, 1)
stocks = ['ORCL', 'TSLA', 'IBM', 'MSFT']

stocksData = pdr.get_data_yahoo(stocks, startDate, endDate)
cleanedData = stocksData['Adj Close']
pctChangeData = cleanedData.pct_change()

covMat = pctChangeData.cov()  #Создаем матрицу ковариации, показывает, как изменение одной акции влияет на другую
covMat = covMat.to_numpy()

n_portfolios = 1000
mean_variance_pairs = []

average_returns = pctChangeData.mean()

for _ in range(n_portfolios):
    weights = np.random.rand(len(stocks)) 
    weights /= np.sum(weights)

    portfolio_E_return = np.dot(weights, average_returns)
    portfolio_E_variance = np.dot(weights.T, np.dot(covMat, weights))

    mean_variance_pairs.append([portfolio_E_return, portfolio_E_variance])

returns, variances = zip(*mean_variance_pairs)
mean_variance_pairs = np.array(mean_variance_pairs)

sorted_portfolios = mean_variance_pairs[mean_variance_pairs[:, 1].argsort()]

efficient_frontier = [sorted_portfolios[0]]
for point in sorted_portfolios:
    if point[0] > efficient_frontier[-1][0]:
        efficient_frontier.append(point)
efficient_frontier = np.array(efficient_frontier)
    
plt.scatter(mean_variance_pairs[:, 1], mean_variance_pairs[:, 0], c='grey', marker='o', s=3)

plt.scatter(efficient_frontier[:, 1], efficient_frontier[:, 0], c='red', marker='o', s=3)

plt.title('Mean-Variance of Portfolios')
plt.xlabel('Variance (Risk)')
plt.ylabel('Expected Return')
plt.grid(True)
plt.show()


