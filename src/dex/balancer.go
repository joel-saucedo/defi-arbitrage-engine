/*
 * Balancer Weighted Pool Mathematical Engine
 * Advanced Go implementation with concurrent processing
 * Optimized for real-time weighted pool arbitrage detection
 */

package balancer

import (
	"errors"
	"math"
	"math/big"
	"sort"
	"sync"
	"time"
)

// Constants for Balancer calculations
const (
	// Maximum number of tokens in a weighted pool
	MaxTokens = 8
	// Minimum weight (1%)
	MinWeight = 0.01
	// Maximum weight (98%)
	MaxWeight = 0.98
	// Swap fee range
	MinSwapFee = 0.0001 // 0.01%
	MaxSwapFee = 0.1    // 10%
	// Precision for fixed-point arithmetic
	Precision = 1e18
	// Exit fee (currently 0 for most pools)
	ExitFee = 0.0
)

// PoolToken represents a token in a Balancer weighted pool
type PoolToken struct {
	Address     string   `json:"address"`
	Symbol      string   `json:"symbol"`
	Decimals    uint8    `json:"decimals"`
	Balance     *big.Int `json:"balance"`
	Weight      float64  `json:"weight"`
	DenormWeight *big.Int `json:"denorm_weight"`
}

// WeightedPool represents a Balancer weighted pool state
type WeightedPool struct {
	Address         string      `json:"address"`
	Tokens          []PoolToken `json:"tokens"`
	SwapFee         float64     `json:"swap_fee"`
	TotalWeight     float64     `json:"total_weight"`
	TotalSupply     *big.Int    `json:"total_supply"`
	LastUpdateBlock uint64      `json:"last_update_block"`
	mutex           sync.RWMutex
}

// SwapParams contains parameters for a swap calculation
type SwapParams struct {
	TokenIn     string   `json:"token_in"`
	TokenOut    string   `json:"token_out"`
	AmountIn    *big.Int `json:"amount_in"`
	MaxPrice    *big.Float `json:"max_price"`
	MaxAmountOut *big.Int `json:"max_amount_out"`
}

// SwapResult contains the result of a swap calculation
type SwapResult struct {
	AmountOut      *big.Int `json:"amount_out"`
	PriceImpact    float64  `json:"price_impact"`
	EffectivePrice float64  `json:"effective_price"`
	SpotPrice      float64  `json:"spot_price"`
	NewBalanceIn   *big.Int `json:"new_balance_in"`
	NewBalanceOut  *big.Int `json:"new_balance_out"`
	TotalFee       *big.Int `json:"total_fee"`
}

// ArbitrageOpportunity represents a profitable arbitrage trade
type ArbitrageOpportunity struct {
	PoolAddress    string   `json:"pool_address"`
	TokenIn        string   `json:"token_in"`
	TokenOut       string   `json:"token_out"`
	AmountIn       *big.Int `json:"amount_in"`
	ExpectedOut    *big.Int `json:"expected_out"`
	ProfitBasisPoints int64 `json:"profit_basis_points"`
	GasCostEstimate   uint64 `json:"gas_cost_estimate"`
	NetProfit      *big.Int `json:"net_profit"`
}

// BalancerCalculator provides mathematical functions for Balancer weighted pools
type BalancerCalculator struct{}

// NewBalancerCalculator creates a new calculator instance
func NewBalancerCalculator() *BalancerCalculator {
	return &BalancerCalculator{}
}

// CalculateSpotPrice computes the spot price between two tokens
// Formula: spotPrice = (balanceIn / weightIn) / (balanceOut / weightOut)
func (bc *BalancerCalculator) CalculateSpotPrice(
	balanceIn *big.Int,
	weightIn float64,
	balanceOut *big.Int,
	weightOut float64,
) *big.Float {
	// Convert to float for calculation
	balInFloat := new(big.Float).SetInt(balanceIn)
	balOutFloat := new(big.Float).SetInt(balanceOut)
	
	// Calculate (balanceIn / weightIn)
	weightInBig := big.NewFloat(weightIn)
	numerator := new(big.Float).Quo(balInFloat, weightInBig)
	
	// Calculate (balanceOut / weightOut)
	weightOutBig := big.NewFloat(weightOut)
	denominator := new(big.Float).Quo(balOutFloat, weightOutBig)
	
	// Return spotPrice = numerator / denominator
	return new(big.Float).Quo(numerator, denominator)
}

// CalculateOutGivenIn computes output amount for exact input
// Formula: amountOut = balanceOut * (1 - (balanceIn / (balanceIn + amountIn * (1 - swapFee)))^(weightIn/weightOut))
func (bc *BalancerCalculator) CalculateOutGivenIn(
	balanceIn *big.Int,
	weightIn float64,
	balanceOut *big.Int,
	weightOut float64,
	amountIn *big.Int,
	swapFee float64,
) *big.Int {
	// Apply swap fee: amountIn * (1 - swapFee)
	feeMultiplier := 1.0 - swapFee
	amountInAfterFee := new(big.Float).SetInt(amountIn)
	amountInAfterFee.Mul(amountInAfterFee, big.NewFloat(feeMultiplier))
	
	// Convert balances to float
	balInFloat := new(big.Float).SetInt(balanceIn)
	balOutFloat := new(big.Float).SetInt(balanceOut)
	
	// Calculate: balanceIn + amountInAfterFee
	denominator := new(big.Float).Add(balInFloat, amountInAfterFee)
	
	// Calculate: balanceIn / (balanceIn + amountInAfterFee)
	ratio := new(big.Float).Quo(balInFloat, denominator)
	
	// Calculate: ratio^(weightIn/weightOut)
	exponent := weightIn / weightOut
	ratioFloat, _ := ratio.Float64()
	powResult := math.Pow(ratioFloat, exponent)
	
	// Calculate: 1 - powResult
	complement := 1.0 - powResult
	
	// Calculate: balanceOut * complement
	result := new(big.Float).Mul(balOutFloat, big.NewFloat(complement))
	
	// Convert back to integer
	resultInt, _ := result.Int(nil)
	return resultInt
}

// CalculateInGivenOut computes input amount for exact output
// Formula: amountIn = (balanceIn * ((balanceOut / (balanceOut - amountOut))^(weightOut/weightIn) - 1)) / (1 - swapFee)
func (bc *BalancerCalculator) CalculateInGivenOut(
	balanceIn *big.Int,
	weightIn float64,
	balanceOut *big.Int,
	weightOut float64,
	amountOut *big.Int,
	swapFee float64,
) *big.Int {
	// Convert to float
	balInFloat := new(big.Float).SetInt(balanceIn)
	balOutFloat := new(big.Float).SetInt(balanceOut)
	amountOutFloat := new(big.Float).SetInt(amountOut)
	
	// Calculate: balanceOut - amountOut
	denominator := new(big.Float).Sub(balOutFloat, amountOutFloat)
	
	// Calculate: balanceOut / (balanceOut - amountOut)
	ratio := new(big.Float).Quo(balOutFloat, denominator)
	
	// Calculate: ratio^(weightOut/weightIn)
	exponent := weightOut / weightIn
	ratioFloat, _ := ratio.Float64()
	powResult := math.Pow(ratioFloat, exponent)
	
	// Calculate: (powResult - 1) * balanceIn
	multiplier := powResult - 1.0
	result := new(big.Float).Mul(balInFloat, big.NewFloat(multiplier))
	
	// Apply swap fee: result / (1 - swapFee)
	feeAdjustment := 1.0 / (1.0 - swapFee)
	result.Mul(result, big.NewFloat(feeAdjustment))
	
	// Convert back to integer
	resultInt, _ := result.Int(nil)
	return resultInt
}

// CalculatePriceImpact computes the price impact of a trade
func (bc *BalancerCalculator) CalculatePriceImpact(
	spotPriceBefore *big.Float,
	spotPriceAfter *big.Float,
) float64 {
	// Calculate: |spotPriceBefore - spotPriceAfter| / spotPriceBefore
	diff := new(big.Float).Sub(spotPriceBefore, spotPriceAfter)
	diff.Abs(diff)
	
	impact := new(big.Float).Quo(diff, spotPriceBefore)
	impactFloat, _ := impact.Float64()
	
	return impactFloat * 100.0 // Return as percentage
}

// NewWeightedPool creates a new weighted pool instance
func NewWeightedPool(address string, swapFee float64) *WeightedPool {
	return &WeightedPool{
		Address:     address,
		Tokens:      make([]PoolToken, 0, MaxTokens),
		SwapFee:     swapFee,
		TotalWeight: 0.0,
		TotalSupply: big.NewInt(0),
		mutex:       sync.RWMutex{},
	}
}

// AddToken adds a token to the weighted pool
func (wp *WeightedPool) AddToken(token PoolToken) error {
	wp.mutex.Lock()
	defer wp.mutex.Unlock()
	
	if len(wp.Tokens) >= MaxTokens {
		return errors.New("pool already has maximum number of tokens")
	}
	
	if token.Weight < MinWeight || token.Weight > MaxWeight {
		return errors.New("token weight out of valid range")
	}
	
	wp.Tokens = append(wp.Tokens, token)
	wp.TotalWeight += token.Weight
	
	return nil
}

// GetTokenIndex returns the index of a token by address
func (wp *WeightedPool) GetTokenIndex(tokenAddress string) int {
	wp.mutex.RLock()
	defer wp.mutex.RUnlock()
	
	for i, token := range wp.Tokens {
		if token.Address == tokenAddress {
			return i
		}
	}
	return -1
}

// SimulateSwap simulates a swap and returns detailed results
func (wp *WeightedPool) SimulateSwap(params SwapParams) (*SwapResult, error) {
	wp.mutex.RLock()
	defer wp.mutex.RUnlock()
	
	// Find token indices
	tokenInIndex := wp.GetTokenIndex(params.TokenIn)
	tokenOutIndex := wp.GetTokenIndex(params.TokenOut)
	
	if tokenInIndex == -1 || tokenOutIndex == -1 {
		return nil, errors.New("token not found in pool")
	}
	
	tokenIn := wp.Tokens[tokenInIndex]
	tokenOut := wp.Tokens[tokenOutIndex]
	
	calc := NewBalancerCalculator()
	
	// Calculate spot price before swap
	spotPriceBefore := calc.CalculateSpotPrice(
		tokenIn.Balance, tokenIn.Weight,
		tokenOut.Balance, tokenOut.Weight,
	)
	
	// Calculate amount out
	amountOut := calc.CalculateOutGivenIn(
		tokenIn.Balance, tokenIn.Weight,
		tokenOut.Balance, tokenOut.Weight,
		params.AmountIn, wp.SwapFee,
	)
	
	// Calculate new balances
	newBalanceIn := new(big.Int).Add(tokenIn.Balance, params.AmountIn)
	newBalanceOut := new(big.Int).Sub(tokenOut.Balance, amountOut)
	
	// Calculate spot price after swap
	spotPriceAfter := calc.CalculateSpotPrice(
		newBalanceIn, tokenIn.Weight,
		newBalanceOut, tokenOut.Weight,
	)
	
	// Calculate price impact
	priceImpact := calc.CalculatePriceImpact(spotPriceBefore, spotPriceAfter)
	
	// Calculate effective price
	amountInFloat := new(big.Float).SetInt(params.AmountIn)
	amountOutFloat := new(big.Float).SetInt(amountOut)
	effectivePrice := new(big.Float).Quo(amountInFloat, amountOutFloat)
	effectivePriceFloat, _ := effectivePrice.Float64()
	
	// Calculate total fee
	feeAmount := new(big.Float).SetInt(params.AmountIn)
	feeAmount.Mul(feeAmount, big.NewFloat(wp.SwapFee))
	totalFee, _ := feeAmount.Int(nil)
	
	// Get spot price as float
	spotPriceFloat, _ := spotPriceBefore.Float64()
	
	return &SwapResult{
		AmountOut:      amountOut,
		PriceImpact:    priceImpact,
		EffectivePrice: effectivePriceFloat,
		SpotPrice:      spotPriceFloat,
		NewBalanceIn:   newBalanceIn,
		NewBalanceOut:  newBalanceOut,
		TotalFee:       totalFee,
	}, nil
}

// BalancerArbitrageScanner scans multiple Balancer pools for arbitrage opportunities
type BalancerArbitrageScanner struct {
	pools map[string]*WeightedPool
	mutex sync.RWMutex
}

// NewBalancerArbitrageScanner creates a new arbitrage scanner
func NewBalancerArbitrageScanner() *BalancerArbitrageScanner {
	return &BalancerArbitrageScanner{
		pools: make(map[string]*WeightedPool),
		mutex: sync.RWMutex{},
	}
}

// AddPool adds a pool to the scanner
func (bas *BalancerArbitrageScanner) AddPool(pool *WeightedPool) {
	bas.mutex.Lock()
	defer bas.mutex.Unlock()
	
	bas.pools[pool.Address] = pool
}

// ScanArbitrageOpportunities scans all pools for arbitrage opportunities
func (bas *BalancerArbitrageScanner) ScanArbitrageOpportunities(
	tokenA, tokenB string,
	amountIn *big.Int,
	minProfitBasisPoints int64,
) []ArbitrageOpportunity {
	bas.mutex.RLock()
	defer bas.mutex.RUnlock()
	
	var opportunities []ArbitrageOpportunity
	var poolPrices []struct {
		address string
		price   *big.Float
	}
	
	// Calculate prices in all pools
	for address, pool := range bas.pools {
		tokenAIndex := pool.GetTokenIndex(tokenA)
		tokenBIndex := pool.GetTokenIndex(tokenB)
		
		if tokenAIndex == -1 || tokenBIndex == -1 {
			continue
		}
		
		tokenAData := pool.Tokens[tokenAIndex]
		tokenBData := pool.Tokens[tokenBIndex]
		
		calc := NewBalancerCalculator()
		price := calc.CalculateSpotPrice(
			tokenAData.Balance, tokenAData.Weight,
			tokenBData.Balance, tokenBData.Weight,
		)
		
		poolPrices = append(poolPrices, struct {
			address string
			price   *big.Float
		}{address, price})
	}
	
	// Sort pools by price
	sort.Slice(poolPrices, func(i, j int) bool {
		return poolPrices[i].price.Cmp(poolPrices[j].price) < 0
	})
	
	// Find arbitrage opportunities between pools with different prices
	for i := 0; i < len(poolPrices); i++ {
		for j := i + 1; j < len(poolPrices); j++ {
			poolLow := bas.pools[poolPrices[i].address]
			poolHigh := bas.pools[poolPrices[j].address]
			
			// Simulate buying from low price pool and selling to high price pool
			opportunity := bas.calculateArbitrageProfit(
				poolLow, poolHigh, tokenA, tokenB, amountIn,
			)
			
			if opportunity != nil && opportunity.ProfitBasisPoints >= minProfitBasisPoints {
				opportunities = append(opportunities, *opportunity)
			}
		}
	}
	
	// Sort by profit descending
	sort.Slice(opportunities, func(i, j int) bool {
		return opportunities[i].ProfitBasisPoints > opportunities[j].ProfitBasisPoints
	})
	
	return opportunities
}

// calculateArbitrageProfit calculates the profit from arbitrage between two pools
func (bas *BalancerArbitrageScanner) calculateArbitrageProfit(
	poolBuy, poolSell *WeightedPool,
	tokenA, tokenB string,
	amountIn *big.Int,
) *ArbitrageOpportunity {
	// Simulate buying tokenB with tokenA in poolBuy
	buyParams := SwapParams{
		TokenIn:  tokenA,
		TokenOut: tokenB,
		AmountIn: amountIn,
	}
	
	buyResult, err := poolBuy.SimulateSwap(buyParams)
	if err != nil {
		return nil
	}
	
	// Simulate selling tokenB for tokenA in poolSell
	sellParams := SwapParams{
		TokenIn:  tokenB,
		TokenOut: tokenA,
		AmountIn: buyResult.AmountOut,
	}
	
	sellResult, err := poolSell.SimulateSwap(sellParams)
	if err != nil {
		return nil
	}
	
	// Calculate profit
	profit := new(big.Int).Sub(sellResult.AmountOut, amountIn)
	
	// Calculate profit in basis points
	profitFloat := new(big.Float).SetInt(profit)
	amountInFloat := new(big.Float).SetInt(amountIn)
	profitRatio := new(big.Float).Quo(profitFloat, amountInFloat)
	profitRatioFloat, _ := profitRatio.Float64()
	profitBasisPoints := int64(profitRatioFloat * 10000)
	
	// Estimate gas cost (approximate for Balancer)
	gasCostEstimate := uint64(200000) // Two swaps plus overhead
	
	return &ArbitrageOpportunity{
		PoolAddress:       poolBuy.Address,
		TokenIn:           tokenA,
		TokenOut:          tokenB,
		AmountIn:          amountIn,
		ExpectedOut:       sellResult.AmountOut,
		ProfitBasisPoints: profitBasisPoints,
		GasCostEstimate:   gasCostEstimate,
		NetProfit:         profit,
	}
}

// GetPoolCount returns the number of pools in the scanner
func (bas *BalancerArbitrageScanner) GetPoolCount() int {
	bas.mutex.RLock()
	defer bas.mutex.RUnlock()
	
	return len(bas.pools)
}

// GetPool returns a pool by address
func (bas *BalancerArbitrageScanner) GetPool(address string) (*WeightedPool, bool) {
	bas.mutex.RLock()
	defer bas.mutex.RUnlock()
	
	pool, exists := bas.pools[address]
	return pool, exists
}
