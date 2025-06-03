# filepath: src/algorithms/amm_optimization.py
"""
advanced automated market maker (amm) curve optimization algorithms
implements mathematical models for:
- constant product (uniswap v2) curve analysis
- concentrated liquidity (uniswap v3) position optimization  
- balancer weighted pool calculations
- curve finance stableswap invariant
- custom bonding curve implementations
"""

import numpy as np
import scipy.optimize as opt
from scipy.integrate import quad
from scipy.special import gamma
from dataclasses import dataclass
from typing import Tuple, List, Optional, Dict
from enum import Enum
import matplotlib.pyplot as plt
from numba import jit, vectorize
import sympy as sp
from web3 import Web3

class PoolType(Enum):
    """amm pool type identification"""
    CONSTANT_PRODUCT = "xy=k"           # uniswap v2
    CONCENTRATED_LIQUIDITY = "cl"       # uniswap v3
    WEIGHTED = "weighted"               # balancer
    STABLE = "stable"                   # curve finance
    BONDING_CURVE = "bonding"          # custom implementations

@dataclass
class PoolState:
    """zero-copy pool state representation"""
    reserve_0: float
    reserve_1: float
    fee_tier: float  # basis points
    sqrt_price_x96: int = 0  # uniswap v3 price encoding
    tick: int = 0
    liquidity: int = 0
    pool_type: PoolType = PoolType.CONSTANT_PRODUCT

@dataclass 
class OptimizationResult:
    """optimization algorithm output"""
    optimal_amount: float
    expected_profit: float
    price_impact: float
    gas_cost_wei: int
    execution_probability: float
    risk_score: float

class ConstantProductOptimizer:
    """uniswap v2 constant product curve optimization
    
    mathematical foundation:
    x * y = k (constant product invariant)
    price = y / x
    slippage = (x * y - (x + dx) * (y - dy)) / (x * y)
    """
    
    @staticmethod
    @jit(nopython=True)
    def calculate_output_amount(reserve_in: float, reserve_out: float, 
                              amount_in: float, fee: float = 0.003) -> float:
        """calculate output amount for given input (with fees)"""
        amount_in_with_fee = amount_in * (1 - fee)
        numerator = amount_in_with_fee * reserve_out
        denominator = reserve_in + amount_in_with_fee
        return numerator / denominator
    
    @staticmethod
    @jit(nopython=True)
    def calculate_price_impact(reserve_in: float, reserve_out: float,
                             amount_in: float) -> float:
        """calculate price impact percentage"""
        price_before = reserve_out / reserve_in
        amount_out = ConstantProductOptimizer.calculate_output_amount(
            reserve_in, reserve_out, amount_in
        )
        price_after = (reserve_out - amount_out) / (reserve_in + amount_in)
        return abs(price_after - price_before) / price_before
    
    def optimize_arbitrage_amount(self, pool_a: PoolState, pool_b: PoolState,
                                gas_price_gwei: float = 20) -> OptimizationResult:
        """find optimal arbitrage amount using mathematical optimization"""
        
        def profit_function(amount: float) -> float:
            # calculate output from pool a
            out_a = self.calculate_output_amount(
                pool_a.reserve_0, pool_a.reserve_1, amount, pool_a.fee_tier / 10000
            )
            
            # calculate output from pool b (reverse swap)
            out_b = self.calculate_output_amount(
                pool_b.reserve_1, pool_b.reserve_0, out_a, pool_b.fee_tier / 10000
            )
            
            # profit = final amount - initial amount - gas costs
            gas_cost_eth = gas_price_gwei * 1e-9 * 150000  # estimated gas
            return out_b - amount - gas_cost_eth
        
        # find optimal amount using scipy optimization
        bounds = [(0.1, min(pool_a.reserve_0, pool_b.reserve_1) * 0.1)]
        result = opt.minimize_scalar(
            lambda x: -profit_function(x), 
            bounds=bounds, 
            method='bounded'
        )
        
        optimal_amount = result.x
        max_profit = -result.fun
        
        # calculate additional metrics
        price_impact_a = self.calculate_price_impact(
            pool_a.reserve_0, pool_a.reserve_1, optimal_amount
        )
        price_impact_b = self.calculate_price_impact(
            pool_b.reserve_1, pool_b.reserve_0, 
            self.calculate_output_amount(pool_a.reserve_0, pool_a.reserve_1, optimal_amount)
        )
        
        return OptimizationResult(
            optimal_amount=optimal_amount,
            expected_profit=max_profit,
            price_impact=max(price_impact_a, price_impact_b),
            gas_cost_wei=int(gas_price_gwei * 1e9 * 150000),
            execution_probability=self._calculate_execution_probability(max_profit),
            risk_score=self._calculate_risk_score(price_impact_a, price_impact_b)
        )
    
    def _calculate_execution_probability(self, profit: float) -> float:
        """estimate execution probability based on profit margin"""
        if profit <= 0:
            return 0.0
        # sigmoid function for probability estimation
        return 1 / (1 + np.exp(-10 * (profit - 0.01)))
    
    def _calculate_risk_score(self, impact_a: float, impact_b: float) -> float:
        """calculate risk score based on price impacts"""
        max_impact = max(impact_a, impact_b)
        # exponential risk scaling
        return min(1.0, max_impact * 10)

class ConcentratedLiquidityOptimizer:
    """uniswap v3 concentrated liquidity optimization
    
    mathematical foundation:
    price = 1.0001^tick
    sqrt_price = sqrt(price) * 2^96
    liquidity calculation for price ranges
    """
    
    def __init__(self):
        self.tick_spacing = {500: 10, 3000: 60, 10000: 200}  # fee tier mappings
    
    @staticmethod
    def tick_to_price(tick: int) -> float:
        """convert tick to price using uniswap v3 formula"""
        return 1.0001 ** tick
    
    @staticmethod
    def price_to_tick(price: float) -> int:
        """convert price to nearest valid tick"""
        return int(np.log(price) / np.log(1.0001))
    
    @staticmethod
    def sqrt_price_to_price(sqrt_price_x96: int) -> float:
        """convert sqrt price (x96) to actual price"""
        sqrt_price = sqrt_price_x96 / (2 ** 96)
        return sqrt_price ** 2
    
    def calculate_liquidity_for_range(self, amount_0: float, amount_1: float,
                                    price_lower: float, price_upper: float,
                                    current_price: float) -> float:
        """calculate liquidity for given price range and amounts"""
        sqrt_price_lower = np.sqrt(price_lower)
        sqrt_price_upper = np.sqrt(price_upper)
        sqrt_price_current = np.sqrt(current_price)
        
        if current_price <= price_lower:
            # all token0
            liquidity = amount_0 / (1/sqrt_price_lower - 1/sqrt_price_upper)
        elif current_price >= price_upper:
            # all token1
            liquidity = amount_1 / (sqrt_price_upper - sqrt_price_lower)
        else:
            # mixed position
            liquidity_0 = amount_0 / (1/sqrt_price_current - 1/sqrt_price_upper)
            liquidity_1 = amount_1 / (sqrt_price_current - sqrt_price_lower)
            liquidity = min(liquidity_0, liquidity_1)
        
        return liquidity
    
    def optimize_position_range(self, current_price: float, volatility: float,
                              capital_amount: float, fee_tier: int) -> Dict:
        """optimize price range for concentrated liquidity position"""
        
        # calculate optimal range based on volatility
        price_std = current_price * volatility
        
        # optimal range bounds (statistical approach)
        lower_bound = current_price - 2 * price_std
        upper_bound = current_price + 2 * price_std
        
        # convert to valid ticks
        tick_lower = self.price_to_tick(lower_bound)
        tick_upper = self.price_to_tick(upper_bound)
        
        # round to valid tick spacing
        spacing = self.tick_spacing.get(fee_tier, 60)
        tick_lower = (tick_lower // spacing) * spacing
        tick_upper = (tick_upper // spacing) * spacing
        
        price_lower = self.tick_to_price(tick_lower)
        price_upper = self.tick_to_price(tick_upper)
        
        # calculate optimal token amounts
        sqrt_ratio = np.sqrt(current_price)
        sqrt_lower = np.sqrt(price_lower)
        sqrt_upper = np.sqrt(price_upper)
        
        # solve for optimal allocation
        delta_1_per_delta_0 = sqrt_ratio - sqrt_lower
        delta_0_per_delta_1 = 1/sqrt_upper - 1/sqrt_ratio
        
        # optimal amounts based on current price
        if delta_1_per_delta_0 > 0:
            amount_1 = capital_amount / (1 + delta_0_per_delta_1 * current_price)
            amount_0 = (capital_amount - amount_1) / current_price
        else:
            amount_0 = capital_amount / current_price
            amount_1 = 0
        
        liquidity = self.calculate_liquidity_for_range(
            amount_0, amount_1, price_lower, price_upper, current_price
        )
        
        return {
            "tick_lower": tick_lower,
            "tick_upper": tick_upper,
            "price_lower": price_lower,
            "price_upper": price_upper,
            "amount_0": amount_0,
            "amount_1": amount_1,
            "liquidity": liquidity,
            "expected_fees": self._estimate_fee_income(liquidity, fee_tier, volatility)
        }
    
    def _estimate_fee_income(self, liquidity: float, fee_tier: int, 
                           volatility: float) -> float:
        """estimate fee income based on liquidity and market conditions"""
        daily_volume_estimate = liquidity * volatility * 10  # heuristic
        fee_rate = fee_tier / 1e6  # convert to decimal
        return daily_volume_estimate * fee_rate

class StableswapOptimizer:
    """curve finance stableswap invariant optimization
    
    mathematical foundation:
    a * n^n * sum(x_i) + d = a * d * n^n + d^(n+1) / (n^n * prod(x_i))
    where a = amplification parameter, d = invariant, n = number of coins
    """
    
    def __init__(self, amplification: float = 100, num_coins: int = 2):
        self.a = amplification
        self.n = num_coins
    
    def calculate_d(self, balances: List[float]) -> float:
        """calculate stableswap invariant d using newton's method"""
        s = sum(balances)
        if s == 0:
            return 0
        
        d = s
        ann = self.a * self.n
        
        for _ in range(255):  # newton iterations
            d_p = d
            for balance in balances:
                d_p = d_p * d // (self.n * balance)
            
            d_prev = d
            d = (ann * s + d_p * self.n) * d // ((ann - 1) * d + (self.n + 1) * d_p)
            
            if abs(d - d_prev) <= 1:
                break
        
        return d
    
    def calculate_y(self, i: int, j: int, x: float, balances: List[float]) -> float:
        """calculate output amount for stableswap"""
        d = self.calculate_d(balances)
        c = d
        s = 0
        ann = self.a * self.n
        
        for k, balance in enumerate(balances):
            if k == i:
                s += x
                c = c * d // (x * self.n)
            elif k != j:
                s += balance
                c = c * d // (balance * self.n)
        
        c = c * d // (ann * self.n)
        b = s + d // ann
        
        y = d
        for _ in range(255):
            y_prev = y
            y = (y * y + c) // (2 * y + b - d)
            if abs(y - y_prev) <= 1:
                break
        
        return y
    
    def optimize_stableswap_arbitrage(self, pool_balances: List[float],
                                    external_prices: List[float]) -> Dict:
        """find optimal arbitrage for stableswap pools"""
        n = len(pool_balances)
        best_profit = 0
        best_trade = None
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                
                # calculate current price in pool
                pool_price = self._get_stableswap_price(i, j, pool_balances)
                external_price = external_prices[j] / external_prices[i]
                
                if pool_price > external_price * 1.001:  # profitable opportunity
                    # find optimal trade size
                    def profit_func(amount):
                        new_balances = pool_balances.copy()
                        new_balances[i] += amount
                        
                        y_out = self.calculate_y(i, j, new_balances[i], new_balances)
                        amount_out = pool_balances[j] - y_out
                        
                        external_value = amount_out * external_prices[j]
                        cost = amount * external_prices[i]
                        
                        return external_value - cost
                    
                    # optimize trade size
                    max_amount = pool_balances[i] * 0.1  # limit to 10% of pool
                    result = opt.minimize_scalar(
                        lambda x: -profit_func(x),
                        bounds=(0.01, max_amount),
                        method='bounded'
                    )
                    
                    if -result.fun > best_profit:
                        best_profit = -result.fun
                        best_trade = {
                            "token_in": i,
                            "token_out": j,
                            "amount_in": result.x,
                            "expected_profit": best_profit,
                            "pool_price": pool_price,
                            "external_price": external_price
                        }
        
        return best_trade or {}
    
    def _get_stableswap_price(self, i: int, j: int, balances: List[float]) -> float:
        """get marginal price between tokens i and j"""
        # small amount for price calculation
        dx = 1e-6
        y_before = self.calculate_y(i, j, balances[i], balances)
        y_after = self.calculate_y(i, j, balances[i] + dx, balances)
        
        dy = y_before - y_after
        return dy / dx if dx > 0 else 0

class BondingCurveOptimizer:
    """custom bonding curve implementations and optimization
    
    supports various curve types:
    - linear: price = m * supply + b
    - exponential: price = a * e^(b * supply)
    - polynomial: price = a * supply^n + b
    - sigmoid: price = a / (1 + e^(-b * (supply - c)))
    """
    
    def __init__(self, curve_type: str = "exponential"):
        self.curve_type = curve_type
    
    def price_function(self, supply: float, params: Dict[str, float]) -> float:
        """calculate token price based on current supply"""
        if self.curve_type == "linear":
            return params["m"] * supply + params["b"]
        elif self.curve_type == "exponential":
            return params["a"] * np.exp(params["b"] * supply)
        elif self.curve_type == "polynomial":
            return params["a"] * (supply ** params["n"]) + params["b"]
        elif self.curve_type == "sigmoid":
            return params["a"] / (1 + np.exp(-params["b"] * (supply - params["c"])))
        else:
            raise ValueError(f"unsupported curve type: {self.curve_type}")
    
    def integral_cost(self, supply_start: float, supply_end: float,
                     params: Dict[str, float]) -> float:
        """calculate total cost to mint tokens (integral of price curve)"""
        def integrand(s):
            return self.price_function(s, params)
        
        cost, _ = quad(integrand, supply_start, supply_end)
        return cost
    
    def optimize_bonding_curve_params(self, target_prices: List[Tuple[float, float]],
                                    curve_type: str = "exponential") -> Dict[str, float]:
        """optimize bonding curve parameters to match target price points"""
        self.curve_type = curve_type
        
        def objective(params_array):
            if curve_type == "exponential":
                params = {"a": params_array[0], "b": params_array[1]}
            elif curve_type == "polynomial":
                params = {"a": params_array[0], "n": params_array[1], "b": params_array[2]}
            else:
                return float('inf')
            
            error = 0
            for supply, target_price in target_prices:
                predicted_price = self.price_function(supply, params)
                error += (predicted_price - target_price) ** 2
            
            return error
        
        # optimization bounds
        if curve_type == "exponential":
            bounds = [(0.01, 100), (0.0001, 1)]  # a, b
            initial_guess = [1, 0.01]
        elif curve_type == "polynomial":
            bounds = [(0.01, 100), (0.1, 5), (0, 100)]  # a, n, b
            initial_guess = [1, 2, 0]
        else:
            raise ValueError(f"optimization not implemented for {curve_type}")
        
        result = opt.minimize(
            objective,
            initial_guess,
            bounds=bounds,
            method='L-BFGS-B'
        )
        
        if curve_type == "exponential":
            return {"a": result.x[0], "b": result.x[1]}
        elif curve_type == "polynomial":
            return {"a": result.x[0], "n": result.x[1], "b": result.x[2]}
    
    def simulate_arbitrage_opportunity(self, current_supply: float,
                                     external_price: float,
                                     params: Dict[str, float]) -> Dict:
        """simulate arbitrage opportunity against bonding curve"""
        curve_price = self.price_function(current_supply, params)
        
        if external_price > curve_price * 1.01:  # buy from curve, sell external
            # find optimal buy amount
            def profit_func(amount):
                new_supply = current_supply + amount
                total_cost = self.integral_cost(current_supply, new_supply, params)
                revenue = amount * external_price
                return revenue - total_cost
            
            # optimize buy amount
            max_amount = current_supply * 0.1  # limit impact
            result = opt.minimize_scalar(
                lambda x: -profit_func(x),
                bounds=(0.01, max_amount),
                method='bounded'
            )
            
            return {
                "direction": "buy_from_curve",
                "amount": result.x,
                "profit": -result.fun,
                "curve_price": curve_price,
                "external_price": external_price
            }
        
        elif external_price < curve_price * 0.99:  # buy external, sell to curve
            # find optimal sell amount
            def profit_func(amount):
                new_supply = current_supply - amount
                revenue = self.integral_cost(new_supply, current_supply, params)
                cost = amount * external_price
                return revenue - cost
            
            max_amount = current_supply * 0.1
            result = opt.minimize_scalar(
                lambda x: -profit_func(x),
                bounds=(0.01, max_amount),
                method='bounded'
            )
            
            return {
                "direction": "sell_to_curve",
                "amount": result.x,
                "profit": -result.fun,
                "curve_price": curve_price,
                "external_price": external_price
            }
        
        return {"direction": "no_arbitrage", "profit": 0}

# testing and demonstration functions
def demonstrate_amm_optimization():
    """demonstrate advanced amm optimization algorithms"""
    print("advanced amm curve optimization demonstration")
    print("=" * 50)
    
    # uniswap v2 optimization
    print("\n1. constant product (uniswap v2) arbitrage optimization")
    optimizer = ConstantProductOptimizer()
    
    pool_a = PoolState(reserve_0=1000000, reserve_1=2000000, fee_tier=30)
    pool_b = PoolState(reserve_0=1000000, reserve_1=2050000, fee_tier=30)
    
    result = optimizer.optimize_arbitrage_amount(pool_a, pool_b)
    print(f"  optimal amount: {result.optimal_amount:.2f}")
    print(f"  expected profit: {result.expected_profit:.6f} eth")
    print(f"  price impact: {result.price_impact:.4f}%")
    print(f"  execution probability: {result.execution_probability:.2f}")
    
    # uniswap v3 position optimization
    print("\n2. concentrated liquidity (uniswap v3) position optimization")
    cl_optimizer = ConcentratedLiquidityOptimizer()
    
    position = cl_optimizer.optimize_position_range(
        current_price=2000,  # eth price
        volatility=0.03,     # 3% daily volatility
        capital_amount=10000,  # $10k position
        fee_tier=3000
    )
    
    print(f"  optimal range: {position['price_lower']:.2f} - {position['price_upper']:.2f}")
    print(f"  liquidity: {position['liquidity']:.2f}")
    print(f"  expected daily fees: ${position['expected_fees']:.2f}")
    
    # stableswap optimization
    print("\n3. stableswap (curve finance) arbitrage optimization")
    stable_optimizer = StableswapOptimizer(amplification=100)
    
    balances = [1000000, 1000000, 1000000]  # 3-pool
    external_prices = [1.0, 1.001, 0.999]   # usdc, usdt, dai
    
    arbitrage = stable_optimizer.optimize_stableswap_arbitrage(balances, external_prices)
    if arbitrage:
        print(f"  token {arbitrage['token_in']} -> token {arbitrage['token_out']}")
        print(f"  amount: {arbitrage['amount_in']:.2f}")
        print(f"  profit: ${arbitrage['expected_profit']:.2f}")
    
    # bonding curve optimization
    print("\n4. bonding curve parameter optimization")
    bonding_optimizer = BondingCurveOptimizer()
    
    # target price points for curve fitting
    target_points = [(100, 1.0), (1000, 10.0), (10000, 100.0)]
    params = bonding_optimizer.optimize_bonding_curve_params(target_points)
    
    print(f"  optimized parameters: {params}")
    
    # simulate arbitrage
    arbitrage_sim = bonding_optimizer.simulate_arbitrage_opportunity(
        current_supply=5000,
        external_price=50,
        params=params
    )
    
    print(f"  arbitrage direction: {arbitrage_sim['direction']}")
    print(f"  potential profit: {arbitrage_sim['profit']:.6f}")

if __name__ == "__main__":
    demonstrate_amm_optimization()
