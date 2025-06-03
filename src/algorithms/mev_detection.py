# filepath: src/algorithms/mev_detection.py
"""
advanced mev (maximal extractable value) detection algorithms
implements real-time mempool analysis for:
- sandwich attack detection and optimization
- frontrunning opportunity identification
- backrunning arbitrage calculations
- liquidation bot strategies
- atomic arbitrage path finding
"""

import asyncio
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Set
from enum import Enum
import networkx as nx
from collections import defaultdict, deque
import heapq
import time
from concurrent.futures import ThreadPoolExecutor
from web3 import Web3
from eth_typing import Address, Hash32

class MEVType(Enum):
    """classification of mev opportunity types"""
    SANDWICH = "sandwich"
    FRONTRUN = "frontrun"
    BACKRUN = "backrun"
    LIQUIDATION = "liquidation"
    ATOMIC_ARBITRAGE = "atomic_arb"
    JAREDFROMSUBWAY = "jfs"  # advanced sandwich variants

@dataclass
class PendingTransaction:
    """mempool transaction representation"""
    hash: str
    from_address: str
    to_address: str
    value: int
    gas_price: int
    gas_limit: int
    input_data: bytes
    nonce: int
    timestamp: float
    decoded_function: Optional[str] = None
    token_addresses: List[str] = None
    amounts: List[int] = None

@dataclass
class MEVOpportunity:
    """identified mev opportunity"""
    type: MEVType
    target_tx: PendingTransaction
    profit_estimate: float
    gas_cost: int
    priority_fee: int
    execution_probability: float
    risk_score: float
    strategy_params: Dict
    
class MempoolAnalyzer:
    """real-time mempool analysis for mev detection"""
    
    def __init__(self, w3: Web3):
        self.w3 = w3
        self.pending_txs: Dict[str, PendingTransaction] = {}
        self.price_cache: Dict[str, float] = {}
        self.gas_tracker = GasTracker()
        self.dex_graph = DEXGraph()
        self.liquidation_monitor = LiquidationMonitor()
        
        # performance optimization
        self.executor = ThreadPoolExecutor(max_workers=8)
        self.analysis_queue = asyncio.Queue(maxsize=1000)
        
    async def start_monitoring(self):
        """begin real-time mempool monitoring"""
        print("[mempool] starting real-time mev detection...")
        
        # start concurrent analysis tasks
        tasks = [
            asyncio.create_task(self._mempool_listener()),
            asyncio.create_task(self._opportunity_analyzer()),
            asyncio.create_task(self._price_updater()),
            asyncio.create_task(self._gas_optimizer())
        ]
        
        await asyncio.gather(*tasks)
    
    async def _mempool_listener(self):
        """listen to pending transactions"""
        while True:
            try:
                # simulate mempool subscription (replace with actual websocket)
                pending_filter = self.w3.eth.filter('pending')
                
                for tx_hash in pending_filter.get_new_entries():
                    try:
                        tx = self.w3.eth.get_transaction(tx_hash)
                        pending_tx = self._parse_transaction(tx)
                        
                        if pending_tx:
                            self.pending_txs[tx_hash.hex()] = pending_tx
                            await self.analysis_queue.put(pending_tx)
                            
                    except Exception as e:
                        continue  # skip failed transactions
                
                await asyncio.sleep(0.1)  # prevent overwhelming
                
            except Exception as e:
                print(f"[mempool error] {e}")
                await asyncio.sleep(1)
    
    async def _opportunity_analyzer(self):
        """analyze transactions for mev opportunities"""
        while True:
            try:
                tx = await self.analysis_queue.get()
                
                # parallel analysis of different mev types
                futures = [
                    self.executor.submit(self._detect_sandwich_opportunity, tx),
                    self.executor.submit(self._detect_frontrun_opportunity, tx),
                    self.executor.submit(self._detect_arbitrage_opportunity, tx),
                    self.executor.submit(self._detect_liquidation_opportunity, tx)
                ]
                
                for future in futures:
                    opportunity = future.result()
                    if opportunity and opportunity.profit_estimate > 0:
                        await self._execute_mev_strategy(opportunity)
                        
            except Exception as e:
                print(f"[analysis error] {e}")
    
    def _parse_transaction(self, tx) -> Optional[PendingTransaction]:
        """parse raw transaction into structured format"""
        try:
            # decode function call if possible
            decoded_function, token_addresses, amounts = self._decode_transaction_data(tx.input)
            
            return PendingTransaction(
                hash=tx.hash.hex(),
                from_address=tx['from'],
                to_address=tx.get('to', ''),
                value=tx.value,
                gas_price=tx.gasPrice,
                gas_limit=tx.gas,
                input_data=tx.input,
                nonce=tx.nonce,
                timestamp=time.time(),
                decoded_function=decoded_function,
                token_addresses=token_addresses,
                amounts=amounts
            )
        except:
            return None
    
    def _decode_transaction_data(self, input_data: bytes) -> Tuple[Optional[str], List[str], List[int]]:
        """decode transaction input data to extract function calls"""
        if len(input_data) < 4:
            return None, [], []
        
        # common dex function signatures
        function_sigs = {
            "0x38ed1739": "swapExactTokensForTokens",
            "0x8803dbee": "swapTokensForExactTokens", 
            "0x7ff36ab5": "swapExactETHForTokens",
            "0x18cbafe5": "swapExactTokensForETH",
            "0xa9059cbb": "transfer",
            "0x095ea7b3": "approve"
        }
        
        sig = input_data[:4].hex()
        function_name = function_sigs.get(sig)
        
        # extract token addresses and amounts (simplified)
        token_addresses = []
        amounts = []
        
        if function_name and "swap" in function_name.lower():
            # decode swap parameters (this is simplified - real implementation 
            # would use proper abi decoding)
            try:
                # extract amounts and token addresses from calldata
                # this is a placeholder - implement proper abi decoding
                pass
            except:
                pass
        
        return function_name, token_addresses, amounts

class SandwichAttackOptimizer:
    """optimize sandwich attack parameters"""
    
    def __init__(self):
        self.min_profit_threshold = 0.01  # minimum profit in eth
        self.max_slippage_tolerance = 0.05  # 5% max slippage
    
    def calculate_sandwich_opportunity(self, target_tx: PendingTransaction,
                                     pool_reserves: Tuple[float, float],
                                     pool_fee: float = 0.003) -> Optional[Dict]:
        """calculate optimal sandwich attack parameters"""
        
        if not target_tx.amounts or len(target_tx.amounts) < 2:
            return None
        
        victim_amount_in = target_tx.amounts[0] / 1e18  # convert from wei
        reserve_in, reserve_out = pool_reserves
        
        # step 1: calculate victim's expected output without sandwich
        victim_output_normal = self._calculate_swap_output(
            reserve_in, reserve_out, victim_amount_in, pool_fee
        )
        
        # step 2: find optimal frontrun amount
        optimal_frontrun = self._optimize_frontrun_amount(
            reserve_in, reserve_out, victim_amount_in, pool_fee
        )
        
        if not optimal_frontrun or optimal_frontrun <= 0:
            return None
        
        # step 3: calculate state after frontrun
        frontrun_output = self._calculate_swap_output(
            reserve_in, reserve_out, optimal_frontrun, pool_fee
        )
        
        new_reserve_in = reserve_in + optimal_frontrun
        new_reserve_out = reserve_out - frontrun_output
        
        # step 4: calculate victim's actual output after frontrun
        victim_output_sandwiched = self._calculate_swap_output(
            new_reserve_in, new_reserve_out, victim_amount_in, pool_fee
        )
        
        # step 5: calculate backrun profit
        final_reserve_in = new_reserve_in + victim_amount_in
        final_reserve_out = new_reserve_out - victim_output_sandwiched
        
        backrun_output = self._calculate_swap_output(
            final_reserve_out, final_reserve_in, frontrun_output, pool_fee
        )
        
        # calculate total profit
        profit = backrun_output - optimal_frontrun
        
        # calculate gas costs
        frontrun_gas = 150000  # estimated gas for swap
        backrun_gas = 150000
        total_gas = frontrun_gas + backrun_gas
        
        if profit > self.min_profit_threshold:
            return {
                "frontrun_amount": optimal_frontrun,
                "expected_profit": profit,
                "victim_slippage": (victim_output_normal - victim_output_sandwiched) / victim_output_normal,
                "gas_cost": total_gas,
                "profit_after_gas": profit - (total_gas * 20e-9),  # assume 20 gwei
                "execution_params": {
                    "frontrun_gas_price": target_tx.gas_price + 1e9,  # +1 gwei
                    "backrun_gas_price": target_tx.gas_price - 1e9   # -1 gwei
                }
            }
        
        return None
    
    def _calculate_swap_output(self, reserve_in: float, reserve_out: float,
                             amount_in: float, fee: float = 0.003) -> float:
        """calculate swap output using constant product formula"""
        amount_in_with_fee = amount_in * (1 - fee)
        numerator = amount_in_with_fee * reserve_out
        denominator = reserve_in + amount_in_with_fee
        return numerator / denominator
    
    def _optimize_frontrun_amount(self, reserve_in: float, reserve_out: float,
                                victim_amount: float, fee: float) -> Optional[float]:
        """find optimal frontrun amount using calculus"""
        
        # objective function: maximize profit from sandwich
        def profit_function(frontrun_amount):
            # frontrun output
            frontrun_out = self._calculate_swap_output(
                reserve_in, reserve_out, frontrun_amount, fee
            )
            
            # new reserves after frontrun
            new_res_in = reserve_in + frontrun_amount
            new_res_out = reserve_out - frontrun_out
            
            # victim's output after frontrun
            victim_out = self._calculate_swap_output(
                new_res_in, new_res_out, victim_amount, fee
            )
            
            # final reserves after victim
            final_res_in = new_res_in + victim_amount
            final_res_out = new_res_out - victim_out
            
            # backrun output
            backrun_out = self._calculate_swap_output(
                final_res_out, final_res_in, frontrun_out, fee
            )
            
            return backrun_out - frontrun_amount
        
        # use golden section search for optimization
        return self._golden_section_search(profit_function, 0.01, reserve_in * 0.1)
    
    def _golden_section_search(self, func, a: float, b: float, tol: float = 1e-6) -> float:
        """golden section search for function optimization"""
        phi = (1 + 5**0.5) / 2  # golden ratio
        c = b - (b - a) / phi
        d = a + (b - a) / phi
        
        while abs(b - a) > tol:
            if func(c) > func(d):
                b = d
            else:
                a = c
            
            c = b - (b - a) / phi
            d = a + (b - a) / phi
        
        return (a + b) / 2

class DEXGraph:
    """graph-based dex liquidity analysis for atomic arbitrage"""
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.pools = {}  # pool_address -> pool_info
        self.token_to_pools = defaultdict(list)
    
    def add_pool(self, pool_address: str, token_a: str, token_b: str,
                reserve_a: float, reserve_b: float, fee: float):
        """add liquidity pool to graph"""
        
        # add edge for token_a -> token_b
        self.graph.add_edge(
            token_a, token_b,
            pool=pool_address,
            reserve_in=reserve_a,
            reserve_out=reserve_b,
            fee=fee,
            weight=-np.log(self._get_exchange_rate(reserve_a, reserve_b))
        )
        
        # add edge for token_b -> token_a
        self.graph.add_edge(
            token_b, token_a,
            pool=pool_address,
            reserve_in=reserve_b,
            reserve_out=reserve_a,
            fee=fee,
            weight=-np.log(self._get_exchange_rate(reserve_b, reserve_a))
        )
        
        self.pools[pool_address] = {
            "token_a": token_a,
            "token_b": token_b,
            "reserve_a": reserve_a,
            "reserve_b": reserve_b,
            "fee": fee
        }
        
        self.token_to_pools[token_a].append(pool_address)
        self.token_to_pools[token_b].append(pool_address)
    
    def find_arbitrage_cycles(self, start_token: str, max_hops: int = 4) -> List[List[str]]:
        """find profitable arbitrage cycles using negative cycle detection"""
        cycles = []
        
        try:
            # use bellman-ford to detect negative cycles
            distances, predecessors = nx.single_source_bellman_ford(
                self.graph, start_token, weight='weight'
            )
            
            # find cycles by checking for improvements
            for token in self.graph.nodes():
                if token in distances:
                    for neighbor in self.graph.neighbors(token):
                        edge_weight = self.graph[token][neighbor]['weight']
                        if distances[token] + edge_weight < distances.get(neighbor, float('inf')):
                            # negative cycle detected
                            cycle = self._extract_cycle(predecessors, token, neighbor)
                            if len(cycle) <= max_hops and cycle[0] == start_token:
                                cycles.append(cycle)
        
        except nx.NetworkXError:
            pass
        
        return cycles
    
    def calculate_cycle_profit(self, cycle: List[str], amount: float) -> float:
        """calculate profit for a given arbitrage cycle"""
        current_amount = amount
        
        for i in range(len(cycle) - 1):
            token_in = cycle[i]
            token_out = cycle[i + 1]
            
            if self.graph.has_edge(token_in, token_out):
                edge_data = self.graph[token_in][token_out]
                reserve_in = edge_data['reserve_in']
                reserve_out = edge_data['reserve_out']
                fee = edge_data['fee']
                
                current_amount = self._calculate_swap_output(
                    reserve_in, reserve_out, current_amount, fee
                )
            else:
                return 0  # no path available
        
        return current_amount - amount  # profit
    
    def _get_exchange_rate(self, reserve_in: float, reserve_out: float) -> float:
        """calculate exchange rate for graph weighting"""
        return reserve_out / reserve_in if reserve_in > 0 else 0
    
    def _extract_cycle(self, predecessors: Dict, start: str, end: str) -> List[str]:
        """extract cycle from predecessor map"""
        cycle = [end]
        current = predecessors.get(end)
        
        while current and current != start and len(cycle) < 10:
            cycle.append(current)
            current = predecessors.get(current)
        
        if current == start:
            cycle.append(start)
            return list(reversed(cycle))
        
        return []
    
    def _calculate_swap_output(self, reserve_in: float, reserve_out: float,
                             amount_in: float, fee: float) -> float:
        """calculate swap output using constant product formula"""
        amount_in_with_fee = amount_in * (1 - fee)
        numerator = amount_in_with_fee * reserve_out
        denominator = reserve_in + amount_in_with_fee
        return numerator / denominator

class LiquidationMonitor:
    """monitor lending protocols for liquidation opportunities"""
    
    def __init__(self):
        self.protocols = {
            "aave": "0x7d2768dE32b0b80b7a3454c06BdAc94A69DDc7A9",
            "compound": "0x3d9819210A31b4961b30EF54bE2aeD79B9c9Cd3B",
            "maker": "0x35D1b3F3D7966A1DFe207aa4514C12a259A0492B"
        }
        self.liquidation_threshold = 0.85  # 85% collateral ratio
    
    async def scan_for_liquidations(self) -> List[Dict]:
        """scan lending protocols for liquidation opportunities"""
        liquidations = []
        
        for protocol_name, protocol_address in self.protocols.items():
            try:
                positions = await self._get_undercollateralized_positions(protocol_address)
                
                for position in positions:
                    liquidation_profit = self._calculate_liquidation_profit(position)
                    
                    if liquidation_profit > 0:
                        liquidations.append({
                            "protocol": protocol_name,
                            "user": position["user"],
                            "collateral_token": position["collateral_token"],
                            "debt_token": position["debt_token"],
                            "collateral_amount": position["collateral_amount"],
                            "debt_amount": position["debt_amount"],
                            "health_factor": position["health_factor"],
                            "estimated_profit": liquidation_profit,
                            "gas_estimate": 300000  # estimated gas for liquidation
                        })
            
            except Exception as e:
                print(f"[liquidation error] {protocol_name}: {e}")
        
        return sorted(liquidations, key=lambda x: x["estimated_profit"], reverse=True)
    
    async def _get_undercollateralized_positions(self, protocol_address: str) -> List[Dict]:
        """get undercollateralized positions from lending protocol"""
        # placeholder - implement actual protocol-specific queries
        return []
    
    def _calculate_liquidation_profit(self, position: Dict) -> float:
        """calculate expected profit from liquidation"""
        # simplified liquidation profit calculation
        collateral_value = position["collateral_amount"] * position.get("collateral_price", 1)
        debt_value = position["debt_amount"] * position.get("debt_price", 1)
        
        liquidation_bonus = 0.05  # 5% liquidation bonus
        profit = collateral_value * liquidation_bonus - debt_value * 0.5  # partial liquidation
        
        return max(0, profit)

class GasTracker:
    """track and optimize gas prices for mev transactions"""
    
    def __init__(self):
        self.base_fee_history = deque(maxsize=100)
        self.priority_fee_history = deque(maxsize=100)
        
    def update_gas_prices(self, base_fee: int, priority_fee: int):
        """update gas price tracking"""
        self.base_fee_history.append(base_fee)
        self.priority_fee_history.append(priority_fee)
    
    def get_optimal_gas_price(self, urgency: str = "medium") -> Tuple[int, int]:
        """get optimal gas price based on current conditions"""
        if not self.base_fee_history:
            return 20e9, 2e9  # default values
        
        base_fee = self.base_fee_history[-1]
        avg_priority = sum(self.priority_fee_history) / len(self.priority_fee_history)
        
        if urgency == "high":
            return int(base_fee * 1.5), int(avg_priority * 2)
        elif urgency == "medium":
            return int(base_fee * 1.2), int(avg_priority * 1.5)
        else:  # low urgency
            return int(base_fee), int(avg_priority)
    
    def estimate_transaction_cost(self, gas_limit: int, urgency: str = "medium") -> int:
        """estimate total transaction cost in wei"""
        base_fee, priority_fee = self.get_optimal_gas_price(urgency)
        return gas_limit * (base_fee + priority_fee)

# demonstration and testing
async def demonstrate_mev_detection():
    """demonstrate mev detection algorithms"""
    print("advanced mev detection demonstration")
    print("=" * 40)
    
    # initialize components
    w3 = Web3(Web3.HTTPProvider("http://localhost:8545"))  # local node
    
    # sandwich attack optimization
    print("\n1. sandwich attack optimization")
    sandwich_optimizer = SandwichAttackOptimizer()
    
    # simulate target transaction
    target_tx = PendingTransaction(
        hash="0x123...",
        from_address="0xabc...",
        to_address="0xdef...",
        value=0,
        gas_price=20e9,
        gas_limit=200000,
        input_data=b"",
        nonce=1,
        timestamp=time.time(),
        amounts=[int(1000e18), 0]  # 1000 tokens
    )
    
    pool_reserves = (1000000, 2000000)  # mock pool reserves
    sandwich_result = sandwich_optimizer.calculate_sandwich_opportunity(
        target_tx, pool_reserves
    )
    
    if sandwich_result:
        print(f"  frontrun amount: {sandwich_result['frontrun_amount']:.2f}")
        print(f"  expected profit: {sandwich_result['expected_profit']:.6f} eth")
        print(f"  victim slippage: {sandwich_result['victim_slippage']:.4f}%")
    
    # dex graph arbitrage
    print("\n2. atomic arbitrage detection")
    dex_graph = DEXGraph()
    
    # add some mock pools
    dex_graph.add_pool("pool1", "WETH", "USDC", 1000, 2000000, 0.003)
    dex_graph.add_pool("pool2", "USDC", "DAI", 2000000, 2000000, 0.001)
    dex_graph.add_pool("pool3", "DAI", "WETH", 2000000, 1000, 0.003)
    
    cycles = dex_graph.find_arbitrage_cycles("WETH", max_hops=4)
    
    for cycle in cycles[:3]:  # show top 3 cycles
        profit = dex_graph.calculate_cycle_profit(cycle, 10)  # 10 eth test
        print(f"  cycle: {' -> '.join(cycle)}")
        print(f"  profit: {profit:.6f} eth")
    
    # liquidation monitoring
    print("\n3. liquidation opportunity scanning")
    liquidation_monitor = LiquidationMonitor()
    
    # this would scan real protocols in production
    print("  scanning aave, compound, maker...")
    print("  [mock] found 3 liquidation opportunities")
    print("  [mock] best profit: 2.5 eth (health factor: 0.82)")

if __name__ == "__main__":
    asyncio.run(demonstrate_mev_detection())
