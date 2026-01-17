#!/bin/bash
# Run Full Demo
# Demonstrates all features of the SDN ML Traffic Management system

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=============================================="
echo "SDN ML Traffic Management - Full Demo"
echo "=============================================="

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_step() { echo -e "${BLUE}[STEP]${NC} $1"; }

# Check Docker
if ! command -v docker &> /dev/null; then
    log_error "Docker is not installed"
    exit 1
fi

# Use docker compose or docker-compose
if docker compose version &> /dev/null 2>&1; then
    COMPOSE_CMD="docker compose"
else
    COMPOSE_CMD="docker-compose"
fi

cd "$PROJECT_DIR/docker"

# Parse arguments
SCENARIO=${1:-"all"}

case $SCENARIO in
    "banking")
        log_step "Running Banking Priority Demo..."
        ;;
    "9am")
        log_step "Running 9AM Burst Demo..."
        ;;
    "all")
        log_step "Running Full Demo..."
        ;;
    *)
        echo "Usage: $0 [banking|9am|all]"
        exit 1
        ;;
esac

# Start services
log_info "Starting services..."
$COMPOSE_CMD up -d

log_info "Waiting for services..."
sleep 15

# Verify services
log_info "Verifying services..."
$COMPOSE_CMD ps

# Run selected scenario
if [ "$SCENARIO" = "banking" ] || [ "$SCENARIO" = "all" ]; then
    log_step "=== Banking Priority Demo ==="
    log_info "Starting bulk transfer (P0)..."
    log_info "Starting banking traffic (P3)..."
    log_info "Watch: banking gets priority, bulk is throttled"
    
    $COMPOSE_CMD exec -T mininet python3 /app/mininet/traffic_profiles.py --scenario banking_priority || true
    
    log_info "Check orchestrator decisions:"
    $COMPOSE_CMD logs --tail=50 ml-orchestrator | grep -E "POLICY|P3|P0" || true
fi

if [ "$SCENARIO" = "9am" ] || [ "$SCENARIO" = "all" ]; then
    log_step "=== 9AM Office Burst Demo ==="
    log_info "Simulating morning login surge..."
    
    $COMPOSE_CMD exec -T mininet python3 /app/mininet/traffic_profiles.py --scenario 9am_burst || true
    
    log_info "Check predictions:"
    $COMPOSE_CMD logs --tail=50 ml-orchestrator | grep -E "predicted|congestion" || true
fi

log_info "Demo complete!"
echo ""
echo "Results:"
echo "  - Orchestrator logs: $COMPOSE_CMD logs ml-orchestrator"
echo "  - Prometheus metrics: http://localhost:9090"
echo ""
echo "To explore further:"
echo "  - Enter Mininet: $COMPOSE_CMD exec mininet bash"
echo "  - Interactive topology: python3 /app/mininet/topo.py"
echo ""
echo "To stop: $COMPOSE_CMD down"
