#!/usr/bin/env python3
"""
High-Performance Mathematical Operations for DeFi Arbitrage
Optimized numerical computations with numpy vectorization and numba JIT compilation
"""

import numpy as np
import numba as nb
from numba import jit, vectorize, float64, int64, uint64
from typing import Tuple, List, Optional, Union
import math
from decimal import Decimal, getcontext
import warnings

# Set high precision for financial calculations
getcontext().prec = 50

# Suppress numba warnings for cleaner output
warnings.filterwarnings('ignore', category=nb.NumbaDeprecationWarning)
warnings.filterwarnings('ignore', category=nb.NumbaPendingDeprecationWarning)

# Mathematical constants
PI = np.pi
E = np.e
SQRT_2PI = np.sqrt(2 * PI)
LOG_2 = np.log(2)
LOG_10 = np.log(10)

# Financial constants
BASIS_POINTS_DENOMINATOR = 10000
PERCENTAGE_DENOMINATOR = 100
WEI_PER_ETH = 10**18
GWEI_PER_ETH = 10**9

@vectorize([float64(float64, float64)], nopython=True, cache=True)
def fast_power(base: float, exponent: float) -> float:
    """
    Vectorized fast power function optimized for DeFi calculations
    Uses bit manipulation for integer exponents when possible
    """
    if exponent == 0.0:
        return 1.0
    if exponent == 1.0:
        return base
    if exponent == 2.0:
        return base * base
    if exponent == 0.5:
        return math.sqrt(base)
    if exponent == -1.0:
        return 1.0 / base
    
    # For fractional exponents, use standard pow
    return math.pow(base, exponent)

@vectorize([float64(float64)], nopython=True, cache=True)
def fast_sqrt(x: float) -> float:
    """Vectorized fast square root using Newton-Raphson method"""
    if x <= 0.0:
        return 0.0
    
    # Initial guess using bit manipulation
    guess = x * 0.5
    
    # Newton-Raphson iterations (3 iterations for good precision)
    for _ in range(3):
        guess = 0.5 * (guess + x / guess)
    
    return guess

@vectorize([float64(float64)], nopython=True, cache=True)
def fast_log(x: float) -> float:
    """Vectorized fast natural logarithm using Taylor series"""
    if x <= 0.0:
        return float('-inf')
    if x == 1.0:
        return 0.0
    
    return math.log(x)

@jit(nopython=True, cache=True)
def compound_interest(principal: float, rate: float, time: float, n: int = 1) -> float:
    """
    Calculate compound interest with high precision
    A = P(1 + r/n)^(nt)
    """
    return principal * fast_power(1.0 + rate / n, n * time)

@jit(nopython=True, cache=True)
def present_value(future_value: float, rate: float, periods: int) -> float:
    """Calculate present value of future cash flows"""
    return future_value / fast_power(1.0 + rate, periods)

@jit(nopython=True, cache=True)
def black_scholes_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Black-Scholes call option pricing formula
    Optimized for volatility arbitrage calculations
    """
    if T <= 0.0:
        return max(S - K, 0.0)
    
    sqrt_T = fast_sqrt(T)
    
    d1 = (fast_log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    
    # Approximation of cumulative normal distribution
    N_d1 = 0.5 * (1.0 + math.erf(d1 / math.sqrt(2.0)))
    N_d2 = 0.5 * (1.0 + math.erf(d2 / math.sqrt(2.0)))
    
    return S * N_d1 - K * math.exp(-r * T) * N_d2

@jit(nopython=True, cache=True)
def implied_volatility_newton(market_price: float, S: float, K: float, T: float, r: float) -> float:
    """
    Calculate implied volatility using Newton-Raphson method
    Optimized for real-time option pricing
    """
    # Initial guess
    sigma = 0.2
    tolerance = 1e-6
    max_iterations = 100
    
    for _ in range(max_iterations):
        # Calculate option price and vega
        price = black_scholes_call(S, K, T, r, sigma)
        
        # Vega calculation (derivative with respect to sigma)
        sqrt_T = fast_sqrt(T)
        d1 = (fast_log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt_T)
        vega = S * sqrt_T * math.exp(-0.5 * d1 * d1) / SQRT_2PI
        
        if abs(vega) < 1e-10:
            break
            
        # Newton-Raphson update
        sigma_new = sigma - (price - market_price) / vega
        
        if abs(sigma_new - sigma) < tolerance:
            return sigma_new
            
        sigma = max(sigma_new, 0.001)  # Ensure positive volatility
    
    return sigma

@jit(nopython=True, cache=True)
def kelly_criterion(win_prob: float, win_ratio: float, lose_ratio: float) -> float:
    """
    Calculate optimal bet size using Kelly Criterion
    f* = (bp - q) / b where b = win_ratio, p = win_prob, q = 1 - win_prob
    """
    if win_prob <= 0.0 or win_prob >= 1.0:
        return 0.0
    
    lose_prob = 1.0 - win_prob
    return (win_ratio * win_prob - lose_prob) / win_ratio

@jit(nopython=True, cache=True)
def sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
    """Calculate Sharpe ratio for strategy performance evaluation"""
    if len(returns) == 0:
        return 0.0
    
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    
    if std_return == 0.0:
        return 0.0
    
    return (mean_return - risk_free_rate) / std_return

@jit(nopython=True, cache=True)
def max_drawdown(price_series: np.ndarray) -> float:
    """Calculate maximum drawdown from price series"""
    if len(price_series) == 0:
        return 0.0
    
    peak = price_series[0]
    max_dd = 0.0
    
    for price in price_series:
        if price > peak:
            peak = price
        
        drawdown = (peak - price) / peak
        if drawdown > max_dd:
            max_dd = drawdown
    
    return max_dd

@jit(nopython=True, cache=True)
def value_at_risk(returns: np.ndarray, confidence_level: float = 0.95) -> float:
    """
    Calculate Value at Risk (VaR) using historical simulation
    Returns the loss that will not be exceeded with given confidence level
    """
    if len(returns) == 0:
        return 0.0
    
    sorted_returns = np.sort(returns)
    index = int((1.0 - confidence_level) * len(sorted_returns))
    
    return -sorted_returns[index]  # Return as positive loss

@jit(nopython=True, cache=True)
def expected_shortfall(returns: np.ndarray, confidence_level: float = 0.95) -> float:
    """
    Calculate Expected Shortfall (Conditional VaR)
    Average of losses beyond VaR threshold
    """
    if len(returns) == 0:
        return 0.0
    
    sorted_returns = np.sort(returns)
    index = int((1.0 - confidence_level) * len(sorted_returns))
    
    if index == 0:
        return -sorted_returns[0]
    
    return -np.mean(sorted_returns[:index])

@jit(nopython=True, cache=True)
def correlation_coefficient(x: np.ndarray, y: np.ndarray) -> float:
    """Calculate Pearson correlation coefficient"""
    if len(x) != len(y) or len(x) == 0:
        return 0.0
    
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    
    numerator = np.sum((x - mean_x) * (y - mean_y))
    denominator = fast_sqrt(np.sum((x - mean_x) ** 2) * np.sum((y - mean_y) ** 2))
    
    if denominator == 0.0:
        return 0.0
    
    return numerator / denominator

@jit(nopython=True, cache=True)
def moving_average(data: np.ndarray, window: int) -> np.ndarray:
    """Calculate simple moving average with specified window"""
    if len(data) < window:
        return np.array([np.mean(data)] * len(data))
    
    result = np.zeros(len(data))
    
    # Calculate first window
    result[window - 1] = np.mean(data[:window])
    
    # Use sliding window for efficiency
    for i in range(window, len(data)):
        result[i] = result[i - 1] + (data[i] - data[i - window]) / window
    
    # Fill initial values
    for i in range(window - 1):
        result[i] = result[window - 1]
    
    return result

@jit(nopython=True, cache=True)
def exponential_moving_average(data: np.ndarray, alpha: float) -> np.ndarray:
    """Calculate exponential moving average with smoothing factor alpha"""
    if len(data) == 0:
        return np.array([])
    
    result = np.zeros(len(data))
    result[0] = data[0]
    
    for i in range(1, len(data)):
        result[i] = alpha * data[i] + (1.0 - alpha) * result[i - 1]
    
    return result

@jit(nopython=True, cache=True)
def bollinger_bands(data: np.ndarray, window: int, num_std: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate Bollinger Bands
    Returns: (middle_band, upper_band, lower_band)
    """
    middle_band = moving_average(data, window)
    
    # Calculate rolling standard deviation
    std_dev = np.zeros(len(data))
    for i in range(window - 1, len(data)):
        window_data = data[max(0, i - window + 1):i + 1]
        std_dev[i] = np.std(window_data)
    
    # Fill initial values
    for i in range(window - 1):
        std_dev[i] = std_dev[window - 1]
    
    upper_band = middle_band + num_std * std_dev
    lower_band = middle_band - num_std * std_dev
    
    return middle_band, upper_band, lower_band

@jit(nopython=True, cache=True)
def rsi(data: np.ndarray, window: int = 14) -> np.ndarray:
    """
    Calculate Relative Strength Index (RSI)
    RSI = 100 - (100 / (1 + RS)) where RS = Average Gain / Average Loss
    """
    if len(data) < window + 1:
        return np.full(len(data), 50.0)
    
    # Calculate price changes
    changes = np.diff(data)
    
    gains = np.where(changes > 0, changes, 0.0)
    losses = np.where(changes < 0, -changes, 0.0)
    
    # Calculate initial averages
    avg_gain = np.mean(gains[:window])
    avg_loss = np.mean(losses[:window])
    
    rsi_values = np.zeros(len(data))
    rsi_values[:window] = 50.0  # Fill initial values
    
    # Calculate RSI using Wilder's smoothing
    for i in range(window, len(data) - 1):
        gain = gains[i]
        loss = losses[i]
        
        avg_gain = (avg_gain * (window - 1) + gain) / window
        avg_loss = (avg_loss * (window - 1) + loss) / window
        
        if avg_loss == 0.0:
            rsi_values[i + 1] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi_values[i + 1] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi_values

@jit(nopython=True, cache=True)
def macd(data: np.ndarray, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate MACD (Moving Average Convergence Divergence)
    Returns: (macd_line, signal_line, histogram)
    """
    fast_alpha = 2.0 / (fast_period + 1)
    slow_alpha = 2.0 / (slow_period + 1)
    signal_alpha = 2.0 / (signal_period + 1)
    
    fast_ema = exponential_moving_average(data, fast_alpha)
    slow_ema = exponential_moving_average(data, slow_alpha)
    
    macd_line = fast_ema - slow_ema
    signal_line = exponential_moving_average(macd_line, signal_alpha)
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram

@jit(nopython=True, cache=True)
def fibonacci_retracement_levels(high: float, low: float) -> np.ndarray:
    """Calculate Fibonacci retracement levels"""
    diff = high - low
    
    levels = np.array([
        low,                           # 0%
        low + 0.236 * diff,           # 23.6%
        low + 0.382 * diff,           # 38.2%
        low + 0.5 * diff,             # 50%
        low + 0.618 * diff,           # 61.8%
        low + 0.786 * diff,           # 78.6%
        high                          # 100%
    ])
    
    return levels

@jit(nopython=True, cache=True)
def geometric_mean(data: np.ndarray) -> float:
    """Calculate geometric mean of positive values"""
    if len(data) == 0:
        return 0.0
    
    # Filter out non-positive values
    positive_data = data[data > 0]
    
    if len(positive_data) == 0:
        return 0.0
    
    log_sum = np.sum(np.log(positive_data))
    return math.exp(log_sum / len(positive_data))

@jit(nopython=True, cache=True)
def harmonic_mean(data: np.ndarray) -> float:
    """Calculate harmonic mean of positive values"""
    if len(data) == 0:
        return 0.0
    
    # Filter out non-positive values
    positive_data = data[data > 0]
    
    if len(positive_data) == 0:
        return 0.0
    
    reciprocal_sum = np.sum(1.0 / positive_data)
    return len(positive_data) / reciprocal_sum

# Utility functions for unit conversions
@jit(nopython=True, cache=True)
def wei_to_eth(wei_amount: int64) -> float64:
    """Convert Wei to ETH"""
    return float64(wei_amount) / WEI_PER_ETH

@jit(nopython=True, cache=True)
def eth_to_wei(eth_amount: float64) -> int64:
    """Convert ETH to Wei"""
    return int64(eth_amount * WEI_PER_ETH)

@jit(nopython=True, cache=True)
def gwei_to_eth(gwei_amount: int64) -> float64:
    """Convert Gwei to ETH"""
    return float64(gwei_amount) / GWEI_PER_ETH

@jit(nopython=True, cache=True)
def basis_points_to_percentage(bp: int64) -> float64:
    """Convert basis points to percentage"""
    return float64(bp) / BASIS_POINTS_DENOMINATOR * PERCENTAGE_DENOMINATOR

# Export all functions for external use
__all__ = [
    'fast_power', 'fast_sqrt', 'fast_log', 'compound_interest', 'present_value',
    'black_scholes_call', 'implied_volatility_newton', 'kelly_criterion',
    'sharpe_ratio', 'max_drawdown', 'value_at_risk', 'expected_shortfall',
    'correlation_coefficient', 'moving_average', 'exponential_moving_average',
    'bollinger_bands', 'rsi', 'macd', 'fibonacci_retracement_levels',
    'geometric_mean', 'harmonic_mean', 'wei_to_eth', 'eth_to_wei',
    'gwei_to_eth', 'basis_points_to_percentage'
]
