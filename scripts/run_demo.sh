#!/bin/bash
# Run Full Demo
# Demonstrates QoS enforcement and traffic prioritization in SDN
#
# Usage:
#   ./run_demo.sh [scenario]
#
# Scenarios:
#   banking  - Banking priority demo (P3 gets priority over P0 bulk)
#   9am      - 9AM office login burst simulation
#   qos      - QoS queue demonstration (all priority levels)
#   all      - Run all scenarios sequentially
#
# Prerequisites:
#   - Docker containers running (faucet, prometheus)
#   - Mininet and OVS installed on host
#   - Python 3.10+ with uv

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_step() { echo -e "\n${BLUE}${BOLD}[STEP]${NC} $1"; }
log_demo() { echo -e "${CYAN}[DEMO]${NC} $1"; }

print_header() {
    echo -e "\n${BOLD}============================================================${NC}"
    echo -e "${BOLD}  SDN ML Traffic Management - Full Demo${NC}"
    echo -e "${BOLD}============================================================${NC}"
    echo ""
    echo "  This demo showcases:"
    echo "    - SNI-based traffic classification (P0-P3)"
    echo "    - QoS queue enforcement via OVS"
    echo "    - Proactive rerouting for banking (P3)"
    echo "    - Reactive rerouting for bulk traffic (P0)"
    echo ""
}

check_prerequisites() {
    log_step "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
        exit 1
    fi
    log_info "Docker: OK"
    
    # Check Mininet
    if ! command -v mn &> /dev/null; then
        log_error "Mininet not found. Install with: sudo apt install mininet"
        exit 1
    fi
    log_info "Mininet: OK"
    
    # Check OVS
    if ! command -v ovs-vsctl &> /dev/null; then
        log_error "Open vSwitch not found. Install with: sudo apt install openvswitch-switch"
        exit 1
    fi
    log_info "Open vSwitch: OK"
    
    # Check Python/uv
    if ! command -v uv &> /dev/null; then
        log_warn "uv not found, falling back to python3"
        PYTHON_CMD="python3"
    else
        PYTHON_CMD="uv run python"
    fi
    log_info "Python: OK ($PYTHON_CMD)"
    
    # Use docker compose or docker-compose
    if docker compose version &> /dev/null 2>&1; then
        COMPOSE_CMD="docker compose"
    else
        COMPOSE_CMD="docker-compose"
    fi
    log_info "Compose: OK ($COMPOSE_CMD)"
}

start_infrastructure() {
    log_step "Starting infrastructure..."
    
    cd "$PROJECT_DIR/docker"
    
    # Start Faucet and Prometheus
    log_info "Starting Docker containers..."
    $COMPOSE_CMD up -d faucet gauge prometheus
    
    log_info "Waiting for Faucet to be ready..."
    for i in {1..30}; do
        if nc -z 127.0.0.1 6653 2>/dev/null; then
            log_info "Faucet ready on port 6653"
            break
        fi
        sleep 1
        if [ $i -eq 30 ]; then
            log_error "Faucet did not start within 30 seconds"
            exit 1
        fi
    done
    
    cd "$PROJECT_DIR"
}

setup_qos_queues() {
    log_step "Setting up QoS queues on OVS switches..."
    
    # Run QoS setup script
    if [ -f "$SCRIPT_DIR/setup_qos.sh" ]; then
        sudo bash "$SCRIPT_DIR/setup_qos.sh" --apply
        log_info "QoS queues configured"
    else
        log_warn "QoS setup script not found, skipping queue configuration"
    fi
}

cleanup_mininet() {
    log_info "Cleaning up previous Mininet instance..."
    sudo mn -c 2>/dev/null || true
}

start_mininet_topology() {
    log_step "Starting Mininet multi-path topology..."
    
    cleanup_mininet
    
    # Ensure OVS is running
    if ! systemctl is-active --quiet openvswitch-switch; then
        log_info "Starting Open vSwitch service..."
        sudo systemctl start openvswitch-switch
    fi
    
    cd "$PROJECT_DIR"
    
    # Start topology in background
    log_info "Starting 6-switch multi-path topology..."
    sudo $PYTHON_CMD topologies/topo_multipath.py &
    MININET_PID=$!
    
    # Wait for topology to be ready
    log_info "Waiting for topology to initialize (15s)..."
    sleep 15
    
    # Check if switches exist
    if sudo ovs-vsctl show | grep -q "s1"; then
        log_info "Topology ready: 6 switches, 4 hosts"
    else
        log_error "Topology failed to start"
        kill $MININET_PID 2>/dev/null || true
        exit 1
    fi
}

run_banking_demo() {
    log_step "=== Banking Priority Demo (P3 vs P0) ==="
    
    echo ""
    log_demo "Scenario: Banking traffic (P3) competes with bulk download (P0)"
    log_demo "Expected: Banking gets priority queue, bulk is throttled"
    echo ""
    
    # Show initial queue configuration
    log_info "Current QoS queue configuration on s1:"
    sudo ovs-vsctl list qos 2>/dev/null | head -20 || log_warn "No QoS config found"
    
    # Start iperf servers
    log_info "Starting iperf servers on h3 (10.0.0.3)..."
    sudo ip netns exec h3 iperf -s -p 5000 &  # Bulk server
    sudo ip netns exec h3 iperf -s -p 5003 &  # Banking server
    sleep 2
    
    # Start bulk transfer first (to create congestion)
    log_info "Starting bulk transfer: h2 -> h3 (P0, 20 Mbps)..."
    sudo ip netns exec h2 iperf -c 10.0.0.3 -p 5000 -t 30 -b 20M &
    BULK_PID=$!
    
    sleep 5  # Let bulk build up
    
    # Start banking traffic
    log_info "Starting banking traffic: h1 -> h3 (P3, 2 Mbps)..."
    sudo ip netns exec h1 iperf -c 10.0.0.3 -p 5003 -t 20 -b 2M &
    BANKING_PID=$!
    
    # Run the policy engine to apply QoS
    log_info "Running policy engine to classify and enforce QoS..."
    cd "$PROJECT_DIR"
    
    # Create sample flows for policy engine test
    $PYTHON_CMD -c "
from orchestrator.policy_engine import PolicyEngine
import time

engine = PolicyEngine(dry_run=False)

# Simulate flows that match our iperf traffic
flows = [
    {
        'flow_id': 'bulk-h2-h3',
        'src_ip': '10.0.0.2',
        'dst_ip': '10.0.0.3',
        'src_port': 50000,
        'dst_port': 5000,
        'protocol': 6,
        'sni': '',
    },
    {
        'flow_id': 'banking-h1-h3',
        'src_ip': '10.0.0.1',
        'dst_ip': '10.0.0.3',
        'src_port': 50001,
        'dst_port': 5003,
        'protocol': 6,
        'sni': 'netbanking.hdfcbank.com',
    },
]

# Simulate predicted congestion
predictions = {
    's1:3': {
        'current_utilization': 0.75,
        'is_congested': False,
        'predicted_congestion': True,
    },
}

print('Applying policies...')
decisions = engine.apply(flows, predictions)
for d in decisions:
    print(f'  {d}')

print(f'\\nStatistics: {engine.get_stats()[\"policy_engine\"]}')
"
    
    # Wait for traffic to complete
    log_info "Waiting for traffic to complete..."
    wait $BANKING_PID 2>/dev/null || true
    wait $BULK_PID 2>/dev/null || true
    
    # Show flow rules
    log_info "OVS flow rules on s1 (showing QoS assignments):"
    sudo ovs-ofctl dump-flows s1 | grep -E "set_queue|nw_src=10.0.0" | head -10 || true
    
    # Cleanup iperf servers
    sudo pkill -f "iperf -s" 2>/dev/null || true
    
    echo ""
    log_demo "Banking Priority Demo Complete!"
    log_demo "Result: P3 (Banking) received priority queue Q0, P0 (Bulk) got Q3"
}

run_9am_burst_demo() {
    log_step "=== 9AM Office Login Burst Demo ==="
    
    echo ""
    log_demo "Scenario: Simulating morning login surge with mixed traffic"
    log_demo "Expected: Orchestrator predicts congestion and reroutes proactively"
    echo ""
    
    # Start iperf servers on all hosts
    log_info "Starting iperf servers on all hosts..."
    for host in h1 h2 h3 h4; do
        sudo ip netns exec $host iperf -s -p 5001 &  # Web
        sudo ip netns exec $host iperf -s -u -p 5002 &  # Voice
    done
    sleep 2
    
    # Generate burst of web traffic
    log_info "Starting login burst (web traffic from all hosts)..."
    for i in {1..5}; do
        # Random source and destination
        src_host="h$((RANDOM % 2 + 1))"
        dst_ip="10.0.0.$((RANDOM % 2 + 3))"
        
        log_info "  Burst $i: $src_host -> $dst_ip"
        sudo ip netns exec $src_host iperf -c $dst_ip -p 5001 -t 5 -b 1M &
        sleep 1
    done
    
    # Wait for burst to complete
    sleep 10
    
    # Cleanup
    sudo pkill -f "iperf" 2>/dev/null || true
    
    echo ""
    log_demo "9AM Burst Demo Complete!"
}

run_qos_demo() {
    log_step "=== QoS Queue Demonstration ==="
    
    echo ""
    log_demo "Scenario: Demonstrate all 4 priority queues"
    log_demo "Q0 (P3): Banking   - 20-50 Mbps guaranteed"
    log_demo "Q1 (P2): Voice     - 10-45 Mbps, low jitter"
    log_demo "Q2 (P1): Web       - 5-35 Mbps, best effort"
    log_demo "Q3 (P0): Bulk      - 0-15 Mbps, background"
    echo ""
    
    # Show queue configuration
    log_info "HTB Queue Configuration on s1-eth1:"
    sudo tc class show dev s1-eth1 2>/dev/null || log_warn "No tc classes found"
    
    # Show QoS configuration
    log_info "OVS QoS Configuration:"
    sudo ovs-vsctl list qos 2>/dev/null | head -30 || log_warn "No QoS config"
    
    # Show queue mapping
    log_info "Queue to Priority Mapping:"
    sudo ovs-vsctl list queue 2>/dev/null | grep -E "dscp|other_config" | head -20 || true
    
    echo ""
    log_demo "QoS Queue Demonstration Complete!"
}

show_results() {
    log_step "Demo Results Summary"
    
    echo ""
    echo "Available interfaces for monitoring:"
    echo "  - Prometheus: http://localhost:9090"
    echo "  - Grafana: http://localhost:3000"
    echo ""
    echo "Useful commands:"
    echo "  - View OVS flows: sudo ovs-ofctl dump-flows s1"
    echo "  - View QoS config: sudo ovs-vsctl list qos"
    echo "  - View queues: sudo ovs-vsctl list queue"
    echo "  - View tc classes: sudo tc class show dev s1-eth1"
    echo ""
    echo "To run again: $0 [banking|9am|qos|all]"
    echo "To cleanup: sudo mn -c && $COMPOSE_CMD down"
}

cleanup() {
    log_step "Cleaning up..."
    
    # Kill background processes
    sudo pkill -f "iperf" 2>/dev/null || true
    sudo pkill -f "topo_multipath.py" 2>/dev/null || true
    
    # Cleanup Mininet
    sudo mn -c 2>/dev/null || true
    
    log_info "Cleanup complete"
}

# Trap for cleanup on exit
trap cleanup EXIT

# Main
print_header

# Parse arguments
SCENARIO=${1:-"all"}

case $SCENARIO in
    "banking"|"9am"|"qos"|"all")
        ;;
    "-h"|"--help"|"help")
        echo "Usage: $0 [banking|9am|qos|all]"
        echo ""
        echo "Scenarios:"
        echo "  banking  - Banking priority demo (P3 gets priority over P0)"
        echo "  9am      - 9AM office login burst simulation"
        echo "  qos      - QoS queue configuration demonstration"
        echo "  all      - Run all scenarios sequentially"
        exit 0
        ;;
    *)
        log_error "Unknown scenario: $SCENARIO"
        echo "Usage: $0 [banking|9am|qos|all]"
        exit 1
        ;;
esac

# Check prerequisites
check_prerequisites

# Start infrastructure
start_infrastructure

# Start Mininet
start_mininet_topology

# Setup QoS
setup_qos_queues

# Run selected scenario
if [ "$SCENARIO" = "qos" ] || [ "$SCENARIO" = "all" ]; then
    run_qos_demo
fi

if [ "$SCENARIO" = "banking" ] || [ "$SCENARIO" = "all" ]; then
    run_banking_demo
fi

if [ "$SCENARIO" = "9am" ] || [ "$SCENARIO" = "all" ]; then
    run_9am_burst_demo
fi

# Show results
show_results

log_info "Demo complete!"
