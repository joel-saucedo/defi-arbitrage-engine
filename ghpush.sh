#!/bin/bash

# ghpush - Automated GitHub repository creation and push script
# Usage: ./ghpush [commit_message]

set -e  # Exit on any error

# Configuration
USERNAME="joel-saucedo"
REPO_NAME="defi-arbitrage-engine"
DEFAULT_COMMIT_MSG="Auto commit $(date '+%Y-%m-%d %H:%M:%S')"

# Get commit message from argument or use default
COMMIT_MSG="${1:-$DEFAULT_COMMIT_MSG}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if GitHub CLI is installed
if ! command -v gh &> /dev/null; then
    print_error "github cli (gh) not found. install required:"
    print_error "visit: https://cli.github.com/"
    exit 1
fi

# Check if user is authenticated with GitHub CLI
if ! gh auth status &> /dev/null; then
    print_warning "github cli authentication required"
    print_status "executing: gh auth login"
    gh auth login
fi

# Check if we're in a git repository
if [ ! -d ".git" ]; then
    print_status "not a git repository. initializing..."
    
    # Check if remote repository exists
    print_status "checking if repository $USERNAME/$REPO_NAME exists on github..."
    
    if gh repo view "$USERNAME/$REPO_NAME" &> /dev/null; then
        print_success "repository already exists on github"
        print_status "cloning existing repository..."
        cd ..
        git clone "https://github.com/$USERNAME/$REPO_NAME.git" temp_clone
        # Move contents to current directory
        mv temp_clone/.git ./
        mv temp_clone/* ./ 2>/dev/null || true
        mv temp_clone/.* ./ 2>/dev/null || true
        rm -rf temp_clone
        print_success "repository cloned and configured"
    else
        print_status "repository doesn't exist. creating new private repo..."
        
        # Initialize git repository
        git init
        
        # Create repository on GitHub (private by default)
        gh repo create "$REPO_NAME" --private --source=. --remote=origin --push=false
        
        # Set up remote
        git remote add origin "https://github.com/$USERNAME/$REPO_NAME.git"
        
        print_success "private repository created: https://github.com/$USERNAME/$REPO_NAME"
    fi
else
    print_status "git repository detected"
fi

# Check if we have any changes to commit
if git diff --quiet && git diff --staged --quiet; then
    # No changes in working directory or staging area
    # Check if we have any commits
    if ! git rev-parse --verify HEAD &> /dev/null; then
        # No commits yet, create initial commit with README
        print_status "no commits detected. creating initial commit..."
        
        if [ ! -f "README.md" ]; then
            cat > README.md << EOF
# $REPO_NAME

defi arbitrage engine - computational trading research laboratory

## overview
experimental platform for mev extraction and arbitrage detection across decentralized exchanges. 
focus on algorithmic optimization, mathematical modeling, and web3 protocol analysis.

### core functionality
- multi-dex arbitrage scanning (uniswap v2/v3, sushiswap, balancer)
- graph theory pathfinding (dijkstra, genetic algorithms)  
- real-time mempool monitoring and tx analysis
- gas optimization and fee prediction models
- sandwich attack detection and protection
- yield farming strategy optimization

### research domains
- **amm mechanics**: constant product, stable swap, concentrated liquidity curves
- **tokenomics**: liquidity incentive modeling and game theory
- **computational methods**: graph algorithms, optimization theory, ml integration
- **mev strategies**: flashloan arbitrage, liquidation bots, priority auctions

### tech stack
- blockchain: web3.py, ethereum, polygon, arbitrum l2s
- algorithms: networkx, numpy, scipy, cvxpy for convex optimization
- machine learning: tensorflow/pytorch for price prediction models
- monitoring: websockets, asyncio, real-time data streams
- visualization: matplotlib, plotly for strategy backtesting

### experimental components
- genetic algorithm route optimization
- reinforcement learning for dynamic strategy adaptation
- zero-knowledge proof integration for private arbitrage
- cross-chain bridge vulnerability analysis
- automated market maker curve research

## development roadmap
1. foundation: web3 integration, basic price oracles
2. optimization: advanced pathfinding, gas fee modeling  
3. intelligence: ml price prediction, strategy automation
4. research: academic paper implementations, novel attack vectors

built on: $(date '+%Y-%m-%d %H:%M:%S')

computational finance research - protocol analysis - algorithmic trading
EOF
        fi
        
        git add .
        git commit -m "init: defi arbitrage research platform setup"
        
        # Push to main branch
        git branch -M main
        git push -u origin main
        
        print_success "initial commit created and pushed"
    else
        # Repository exists but no changes to commit
        print_warning "no changes detected. logging contribution timestamp..."
        
        # Create a contributions log file
        echo "contribution logged: $(date '+%Y-%m-%d %H:%M:%S')" >> contributions.log
        git add contributions.log
        git commit -m "$COMMIT_MSG"
        git push
        
        print_success "contribution timestamp logged and pushed"
    fi
else
    # We have changes to commit
    print_status "changes detected. staging all files..."
    git add .
    
    print_status "committing: '$COMMIT_MSG'"
    git commit -m "$COMMIT_MSG"
    
    print_status "pushing to remote..."
    git push
    
    print_success "changes committed and pushed successfully"
fi

# Show repository status
print_status "repository status:"
echo "repo: https://github.com/$USERNAME/$REPO_NAME"
echo "visibility: public"
echo "latest commit: $COMMIT_MSG"
echo "timestamp: $(date '+%Y-%m-%d %H:%M:%S')"

print_success "ghpush execution complete"
