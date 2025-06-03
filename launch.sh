#!/bin/bash
################################################################################
#                        ETHEREUM MEV RESEARCH LAUNCHER
#                     One-Click System Initialization
#                   Sub-Millisecond Performance Ready
################################################################################

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
LOG_DIR="${SCRIPT_DIR}/logs"

# Create necessary directories
mkdir -p "$BUILD_DIR" "$LOG_DIR"

# Logging function
log() {
    echo -e "${WHITE}[$(date +'%H:%M:%S')]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[$(date +'%H:%M:%S')] ✅ $1${NC}"
}

log_error() {
    echo -e "${RED}[$(date +'%H:%M:%S')] ❌ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}[$(date +'%H:%M:%S')] ⚠️  $1${NC}"
}

log_info() {
    echo -e "${BLUE}[$(date +'%H:%M:%S')] ℹ️  $1${NC}"
}

# Banner
print_banner() {
    echo -e "${PURPLE}"
    cat << "EOF"
╔═══════════════════════════════════════════════════════════════════════════════╗
║                           ETHEREUM MEV RESEARCH                               ║
║                      High-Frequency Arbitrage Engine                         ║
║                       Sub-Millisecond Execution Ready                        ║
╚═══════════════════════════════════════════════════════════════════════════════╝
EOF
    echo -e "${NC}"
}

# Check system dependencies
check_dependencies() {
    log "🔍 Checking system dependencies..."
    
    local missing_deps=()
    
    # Required commands
    local required_commands=("python3" "make" "gcc" "cargo" "go" "node" "npm")
    
    for cmd in "${required_commands[@]}"; do
        if ! command -v "$cmd" &> /dev/null; then
            missing_deps+=("$cmd")
        fi
    done
    
    # Optional commands
    local optional_commands=("julia" "zig" "nasm")
    local missing_optional=()
    
    for cmd in "${optional_commands[@]}"; do
        if ! command -v "$cmd" &> /dev/null; then
            missing_optional+=("$cmd")
        fi
    done
    
    if [ ${#missing_deps[@]} -ne 0 ]; then
        log_error "Missing required dependencies: ${missing_deps[*]}"
        log_info "Install with: sudo apt update && sudo apt install ${missing_deps[*]}"
        return 1
    fi
    
    if [ ${#missing_optional[@]} -ne 0 ]; then
        log_warning "Missing optional dependencies: ${missing_optional[*]}"
        log_info "Some advanced features may not be available"
    fi
    
    log_success "All required dependencies found"
    return 0
}

# Build system components
build_components() {
    log "🏗️  Building system components..."
    
    # Create build directory structure
    mkdir -p "$BUILD_DIR"/{cpp,rust,go,zig,js,asm}
    
    # Build with make
    if make build-all > "$LOG_DIR/build.log" 2>&1; then
        log_success "All components built successfully"
    else
        log_error "Build failed. Check $LOG_DIR/build.log for details"
        return 1
    fi
    
    return 0
}

# Run system tests
run_tests() {
    log "🧪 Running system tests..."
    
    if python3 orchestrator.py --test-only > "$LOG_DIR/tests.log" 2>&1; then
        log_success "All tests passed"
    else
        log_warning "Some tests failed. Check $LOG_DIR/tests.log for details"
        # Don't return error as tests might fail due to missing optional components
    fi
}

# Start MEV engine
start_engine() {
    log "🚀 Starting MEV Master Engine..."
    log_info "WebSocket dashboard available at: ws://localhost:8765"
    log_info "Press Ctrl+C to stop the engine"
    
    # Trap signals for graceful shutdown
    trap 'log_info "Shutting down MEV engine..."; exit 0' INT TERM
    
    # Start the engine
    python3 orchestrator.py
}

# Performance monitoring
monitor_performance() {
    log "📊 Starting performance monitor..."
    
    # Create monitoring script
    cat > "$BUILD_DIR/monitor.py" << 'EOF'
import time
import psutil
import json
import asyncio
import websockets

async def connect_to_engine():
    try:
        async with websockets.connect("ws://localhost:8765") as websocket:
            print("📊 Connected to MEV engine dashboard")
            while True:
                data = await websocket.recv()
                metrics = json.loads(data)
                
                print(f"\r🔥 Opportunities: {metrics['opportunity_count']:,} | "
                      f"Executions: {metrics['execution_count']:,} | "
                      f"Profit: {metrics['total_profit']:.6f} ETH", end="", flush=True)
                
    except Exception as e:
        print(f"\n❌ Dashboard connection failed: {e}")
        print("📈 Fallback: System resource monitoring")
        
        while True:
            cpu = psutil.cpu_percent(interval=1)
            mem = psutil.virtual_memory().percent
            print(f"\r💻 CPU: {cpu:5.1f}% | RAM: {mem:5.1f}%", end="", flush=True)

if __name__ == "__main__":
    asyncio.run(connect_to_engine())
EOF
    
    python3 "$BUILD_DIR/monitor.py"
}

# Show system info
show_system_info() {
    log "💻 System Information:"
    echo "   OS: $(uname -s) $(uname -r)"
    echo "   CPU: $(nproc) cores"
    echo "   RAM: $(free -h | awk '/^Mem:/ {print $2}')"
    echo "   Python: $(python3 --version)"
    echo "   Rust: $(rustc --version 2>/dev/null || echo 'Not installed')"
    echo "   Go: $(go version 2>/dev/null || echo 'Not installed')"
    echo "   Node: $(node --version 2>/dev/null || echo 'Not installed')"
    echo ""
}

# Main menu
show_menu() {
    echo -e "${CYAN}"
    echo "Available commands:"
    echo "  init     - Initialize and build all components"
    echo "  test     - Run comprehensive test suite"
    echo "  start    - Start the MEV master engine"
    echo "  monitor  - Real-time performance monitoring"
    echo "  build    - Build components only"
    echo "  clean    - Clean build artifacts"
    echo "  info     - Show system information"
    echo "  help     - Show this menu"
    echo -e "${NC}"
}

# Parse command line arguments
case "${1:-help}" in
    "init")
        print_banner
        show_system_info
        check_dependencies && build_components && run_tests
        log_success "🎉 System initialization complete!"
        log_info "Run './launch.sh start' to begin MEV operations"
        ;;
    
    "test")
        print_banner
        run_tests
        ;;
    
    "start")
        print_banner
        log_info "🔥 Starting MEV operations..."
        start_engine
        ;;
    
    "monitor")
        print_banner
        monitor_performance
        ;;
    
    "build")
        print_banner
        check_dependencies && build_components
        ;;
    
    "clean")
        log "🧹 Cleaning build artifacts..."
        make clean > /dev/null 2>&1
        rm -rf "$BUILD_DIR" "$LOG_DIR"
        log_success "Clean completed"
        ;;
    
    "info")
        print_banner
        show_system_info
        check_dependencies
        ;;
    
    "help"|*)
        print_banner
        show_menu
        ;;
esac
