package main

import "C"
import (
"math"
"runtime"
"sync"
"time"
)

//export calculateSwapAmount
func calculateSwapAmount(balanceIn, weightIn, balanceOut, weightOut, amountIn, swapFee float64) float64 {
adjustedAmountIn := amountIn * (1.0 - swapFee)
base := balanceIn / (balanceIn + adjustedAmountIn)
exponent := weightIn / weightOut

if base <= 0 || exponent <= 0 {
return 0
}

return balanceOut * (1.0 - math.Pow(base, exponent))
}

//export benchmarkEngine
func benchmarkEngine(iterations int) float64 {
start := time.Now()

for i := 0; i < iterations; i++ {
calculateSwapAmount(1000000.0, 0.5, 2000000.0, 0.5, 1000.0, 0.003)
}

return time.Since(start).Seconds()
}

func main() {}
