import numpy as np
import pandas as pd
from pandas_datareader import data as pdr
import yfinance as yf
import matplotlib.pyplot as plt
from typing import List, Tuple


yf.pdr_override()


def getData(stocks: List[str], start: str, end: str) -> Tuple[pd.Series, pd.DataFrame]:
    """Ф-ия собирает данные о котировках акций на американском рынке

    Args:
        stocks (List[str]): Список тикеров
        start (str): Дата начала отсчета
        end (str): Дата конца отсчета

    Returns:
        Tuple[pd.Series, pd.DataFrame]: Возвращает среднюю доходность и ковариационную матрицу
    """
    stockData = pdr.get_data_yahoo(stocks, start=start, end=end)
    stockData = stockData['Adj Close']
    returns = stockData.pct_change()
    meanReturns = returns.mean()
    covMatrix = returns.cov()
    return meanReturns, covMatrix

def portfolioPerformance(weights: List[float], meanReturns: pd.Series, covMatrix: pd.DataFrame) -> Tuple[float, float]:
    """Ф-ия считает доходность и риск портфеля

    Args:
        weights (List[float]): Веса для каждой акции в портфеле
        meanReturns (pd.Series): Средняя доходность акций
        covMatrix (pd.DataFrame): Ковариационная матрица

    Returns:
        Tuple[float, float]: Возвращает доходность и риск портфеля
    """
    returns = np.sum(meanReturns * weights) * 252
    std = np.sqrt(np.dot(weights, np.dot(covMatrix, weights))) * np.sqrt(252)
    return returns * 100, std * 100

def efficientFrontier(meanReturns: pd.Series, covMatrix: pd.DataFrame, numPortfolios: int, riskFreeRate: float) -> Tuple[np.ndarray, List[np.ndarray], float, np.ndarray]:
    """Ф-ия строит эффективный фронт и находит портфель с максимльным Sharpe Ratio при заданном безрисковом активе

    Args:
        meanReturns (pd.Series): Средняя доходность акций
        covMatrix (pd.DataFrame): Ковариационная матрица
        numPortfolios (int): Количество портфелей для генерации
        riskFreeRate (float): Безрисковая ставка(в процентах)

    Returns:
        Tuple[np.ndarray, List[np.ndarray], float, np.ndarray]: Возвращает эффективный фронт, веса портфелей, максимальный Sharpe Ratio и веса портфеля с максимальным Sharpe Ratio
    """
    results = np.zeros((3, numPortfolios))
    weights_record = []
    max_sharpe_ratio = -1
    max_sharpe_weights = np.array([])

    for i in range(numPortfolios):
        weights = np.random.random(len(meanReturns))
        weights /= np.sum(weights)
        weights_record.append(weights)
        portfolio_return, portfolio_std = portfolioPerformance(weights, meanReturns, covMatrix)
        sharpe_ratio = (portfolio_return - riskFreeRate) / portfolio_std

        if sharpe_ratio > max_sharpe_ratio:
            max_sharpe_ratio = sharpe_ratio
            max_sharpe_weights = weights

        results[0, i] = portfolio_std
        results[1, i] = portfolio_return
        results[2, i] = sharpe_ratio

    return results, weights_record, max_sharpe_ratio, max_sharpe_weights


def plot_efficient_frontier(results: np.ndarray, stocks: List[str], max_sr_volatility: float, max_sr_return: float, max_SR_weights: np.ndarray, riskFreeRate: float) -> None:
    """_summary_

    Args:
        results (np.ndarray): Наиболее эффективные порфтели
        stocks (List[str]): Список акций
        max_sr_volatility (float): Волатильность портфеля с максимальным Sharpe ratio
        max_sr_return (float): Доходность портфеля с максимальным Sharpe Ratio
        max_SR_weights (np.ndarray): Распределение весов в портфеле с максимальным Sharpe Ratio
        riskFreeRate (float): Безрисковая ставка
    """
    annotation_text = 'Max Sharpe Ratio Weights:\n' + '\n'.join([f'{stock}: {weight:.2f}' for stock, weight in zip(stocks, max_SR_weights)])

    plt.annotate(annotation_text, (0,0), color='blue', xytext=(20,10), textcoords="offset points")

    slope = (max_sr_return - riskFreeRate) / max_sr_volatility
    line_x = np.linspace(0, max_sr_volatility, 100)
    line_y = riskFreeRate + slope * line_x
    plt.plot(line_x, line_y, color='green', linestyle='--')

    plt.scatter(results[0,:], results[1,:], c=results[2,:], cmap='YlGnBu', marker='o')
    plt.scatter(max_sr_volatility, max_sr_return, color='red', marker="*", s=100)

    plt.title('Efficient Frontier')
    plt.xlabel('Volatility (%)')
    plt.ylabel('Expected Returns (%)')
    plt.colorbar(label='Sharpe Ratio')
    plt.xlim(0, 30)
    plt.ylim(0, 30)
    plt.show()

if __name__ == "__main__":
    stocks_input = [stock.strip() for stock in input("Введите список акций через запятую (например: AAPL, MSFT, GOOG): ").split(',')]
    start_date = input("Введите начальную дату (например: 2010-01-01): ")
    end_date = input("Введите конечную дату (например: 2020-01-01): ")
    numPortfolios = int(input("Введите количество портфелей для анализа: "))
    riskFreeRate = float(input("Введите безрисковую процентную ставку (например, 25): "))

    meanReturns, covMatrix = getData(stocks_input, start_date, end_date)
    results, weights, max_SR, max_SR_weights = efficientFrontier(meanReturns, covMatrix, numPortfolios, riskFreeRate)
    max_sr_return, max_sr_volatility = portfolioPerformance(max_SR_weights, meanReturns, covMatrix)
    plot_efficient_frontier(results, stocks_input, max_sr_volatility, max_sr_return, max_SR_weights, riskFreeRate)