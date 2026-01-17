#!/bin/bash
# Run MVP Demo
# Tests the minimal end-to-end pipeline

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=================================="
echo "SDN ML Traffic Management - MVP Demo"
echo "=================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check Docker
if ! command -v docker &> /dev/null; then
    log_error "Docker is not installed"
    exit 1
fi

if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    log_error "Docker Compose is not installed"
    exit 1
fi

# Use docker compose or docker-compose
if docker compose version &> /dev/null 2>&1; then
    COMPOSE_CMD="docker compose"
else
    COMPOSE_CMD="docker-compose"
fi

cd "$PROJECT_DIR/docker"

log_info "Starting services..."
$COMPOSE_CMD up -d

log_info "Waiting for services to be ready..."
sleep 10

# Check Faucet health
log_info "Checking Faucet..."
if curl -s http://localhost:9302/metrics > /dev/null 2>&1; then
    log_info "Faucet is running"
else
    log_warn "Faucet metrics not available yet"
fi

# Check Prometheus health
log_info "Checking Prometheus..."
if curl -s http://localhost:9090/-/healthy > /dev/null 2>&1; then
    log_info "Prometheus is healthy"
else
    log_warn "Prometheus not available yet"
fi

# Start Mininet topology
log_info "Starting Mininet topology..."
$COMPOSE_CMD exec -T mininet bash -c "
    # Start OVS if not running
    /usr/local/bin/start-ovs.sh &
    sleep 2
    
    # Run topology (non-interactive for demo)
    python3 /app/mininet/topo.py &
    sleep 5
    
    echo 'Topology started'
"

# Generate traffic
log_info "Generating test traffic..."
$COMPOSE_CMD exec -T mininet bash -c "
    # Start iperf server on h2
    mn -c 'h2 iperf -s -p 5001 &'
    sleep 1
    
    # Run iperf client on h1
    mn -c 'h1 iperf -c 10.0.0.2 -p 5001 -t 10 -i 1'
" || log_warn "Traffic generation skipped (topology may not be ready)"

# Check orchestrator logs
log_info "Checking orchestrator..."
$COMPOSE_CMD logs --tail=20 ml-orchestrator

log_info "MVP demo complete!"
echo ""
echo "Services are running. You can:"
echo "  - View Prometheus: http://localhost:9090"
echo "  - View Faucet metrics: http://localhost:9302/metrics"
echo "  - Enter Mininet: $COMPOSE_CMD exec mininet bash"
echo "  - View orchestrator logs: $COMPOSE_CMD logs -f ml-orchestrator"
echo ""
echo "To stop: $COMPOSE_CMD down"
