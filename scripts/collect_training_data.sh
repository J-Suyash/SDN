#!/bin/bash
# Collect Training Data
# Runs traffic profiles and collects telemetry for ML training

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=============================================="
echo "SDN ML - Training Data Collection"
echo "=============================================="

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }

# Use docker compose or docker-compose
if docker compose version &> /dev/null 2>&1; then
    COMPOSE_CMD="docker compose"
else
    COMPOSE_CMD="docker-compose"
fi

cd "$PROJECT_DIR/docker"

# Configuration
DURATION=${1:-300}  # Default 5 minutes
OUTPUT_DIR="$PROJECT_DIR/data/processed"

log_info "Data collection duration: ${DURATION}s"
log_info "Output directory: $OUTPUT_DIR"

# Ensure services are running
log_info "Checking services..."
$COMPOSE_CMD up -d

# Wait for services
sleep 10

# Start data collection in orchestrator
log_info "Starting data collection..."

# Run mixed traffic profiles
log_info "Generating mixed traffic profiles..."

for i in $(seq 1 $((DURATION / 60))); do
    log_info "Iteration $i - Running traffic profiles..."
    
    # Banking traffic
    $COMPOSE_CMD exec -T mininet bash -c "
        # P3 Banking
        iperf -c 10.0.0.2 -p 5003 -t 10 -b 1M &
        # P2 Voice
        iperf -c 10.0.0.2 -u -p 5002 -t 30 -b 64k &
        # P1 Web
        iperf -c 10.0.0.2 -p 5001 -t 20 -b 2M &
        # P0 Bulk
        iperf -c 10.0.0.2 -p 5000 -t 60 -b 8M &
        wait
    " 2>/dev/null || log_warn "Traffic generation partial"
    
    log_info "Waiting for next iteration..."
    sleep 60
done

log_info "Data collection complete!"
echo ""
echo "Generated datasets:"
echo "  - Flows: $OUTPUT_DIR/flows.csv"
echo "  - Links: $OUTPUT_DIR/link_timeseries.csv"
echo ""
echo "Next steps:"
echo "  1. Upload datasets to Google Colab"
echo "  2. Run training notebooks"
echo "  3. Download trained models to ml/models/"
