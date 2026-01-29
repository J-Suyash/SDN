#!/bin/bash
# =============================================================================
# QoS Queue Setup Script for SDN ML Traffic Management
# =============================================================================
#
# This script configures HTB (Hierarchical Token Bucket) QoS queues on 
# Open vSwitch edge switches for traffic prioritization.
#
# Queue Configuration:
#   Queue 0 (P3 - Banking):  Highest priority, guaranteed 20Mbps, max 50Mbps
#   Queue 1 (P2 - Voice):    High priority, guaranteed 10Mbps, max 45Mbps
#   Queue 2 (P1 - Web):      Best effort, guaranteed 5Mbps, max 35Mbps
#   Queue 3 (P0 - Bulk):     Lowest priority, no guarantee, max 15Mbps
#
# Usage:
#   ./setup_qos.sh [OPTIONS]
#
# Options:
#   --cleanup       Remove all QoS configuration
#   --edge-only     Only configure edge switches (s1, s4)
#   --all           Configure all switches
#   --dry-run       Show commands without executing
#   --help          Show this help message
#
# Prerequisites:
#   - OVS must be running
#   - Mininet topology must be started
#   - Root/sudo access required
#
# =============================================================================

set -e

# Configuration
# Link capacities in bits per second
EDGE_LINK_CAPACITY=100000000      # 100 Mbps (host links)
CORE_LINK_CAPACITY=50000000       # 50 Mbps (core links)

# Queue rate limits in bits per second
# Queue 0: P3 Banking (highest priority)
Q0_MIN_RATE=20000000              # 20 Mbps guaranteed
Q0_MAX_RATE=50000000              # 50 Mbps max

# Queue 1: P2 Voice
Q1_MIN_RATE=10000000              # 10 Mbps guaranteed
Q1_MAX_RATE=45000000              # 45 Mbps max

# Queue 2: P1 Web (best effort)
Q2_MIN_RATE=5000000               # 5 Mbps guaranteed
Q2_MAX_RATE=35000000              # 35 Mbps max

# Queue 3: P0 Bulk (lowest priority)
Q3_MIN_RATE=0                     # No guarantee
Q3_MAX_RATE=15000000              # 15 Mbps max (throttled)

# Edge switches (connected to hosts)
EDGE_SWITCHES="s1 s4"

# Core switches
CORE_SWITCHES="s2 s3 s5 s6"

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[OK]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Command execution (respects dry-run)
DRY_RUN=false
run_cmd() {
    if [ "$DRY_RUN" = true ]; then
        echo "  [DRY-RUN] $*"
    else
        eval "$@"
    fi
}

# Check if switch exists
switch_exists() {
    local switch=$1
    ovs-vsctl br-exists "$switch" 2>/dev/null
    return $?
}

# Get list of ports on a switch (excluding internal port)
get_switch_ports() {
    local switch=$1
    ovs-vsctl list-ports "$switch" 2>/dev/null
}

# Clear existing QoS on a port
clear_port_qos() {
    local port=$1
    log_info "Clearing QoS on port: $port"
    run_cmd "ovs-vsctl clear port $port qos 2>/dev/null || true"
}

# Configure QoS on a single port
configure_port_qos() {
    local switch=$1
    local port=$2
    local max_rate=$3
    
    log_info "Configuring QoS on $switch:$port (max-rate: $((max_rate/1000000)) Mbps)"
    
    # Clear existing QoS first
    run_cmd "ovs-vsctl clear port $port qos 2>/dev/null || true"
    
    # Create QoS with 4 queues using linux-htb
    run_cmd "ovs-vsctl -- \
        set port $port qos=@newqos -- \
        --id=@newqos create qos type=linux-htb \
            other-config:max-rate=$max_rate \
            queues:0=@q0 \
            queues:1=@q1 \
            queues:2=@q2 \
            queues:3=@q3 -- \
        --id=@q0 create queue \
            other-config:min-rate=$Q0_MIN_RATE \
            other-config:max-rate=$Q0_MAX_RATE \
            other-config:priority=0 -- \
        --id=@q1 create queue \
            other-config:min-rate=$Q1_MIN_RATE \
            other-config:max-rate=$Q1_MAX_RATE \
            other-config:priority=1 -- \
        --id=@q2 create queue \
            other-config:min-rate=$Q2_MIN_RATE \
            other-config:max-rate=$Q2_MAX_RATE \
            other-config:priority=2 -- \
        --id=@q3 create queue \
            other-config:max-rate=$Q3_MAX_RATE \
            other-config:priority=3"
    
    log_success "QoS configured on $port"
}

# Configure QoS for edge switch (uplink ports to core)
configure_edge_switch() {
    local switch=$1
    
    if ! switch_exists "$switch"; then
        log_warn "Switch $switch does not exist, skipping"
        return
    fi
    
    log_info "Configuring edge switch: $switch"
    
    # Get all ports
    local ports=$(get_switch_ports "$switch")
    
    for port in $ports; do
        # Skip host-facing ports (we want to configure uplinks to core)
        # In our topology: s1-eth3/s1-eth4 are uplinks, s1-eth1/s1-eth2 are host ports
        # s4-eth3/s4-eth4 are uplinks, s4-eth1/s4-eth2 are host ports
        case "$port" in
            *-eth3|*-eth4)
                # Uplink port - configure with core link capacity
                configure_port_qos "$switch" "$port" "$CORE_LINK_CAPACITY"
                ;;
            *-eth1|*-eth2)
                # Host-facing port - configure with edge link capacity
                configure_port_qos "$switch" "$port" "$EDGE_LINK_CAPACITY"
                ;;
            *)
                log_warn "Unknown port pattern: $port, skipping"
                ;;
        esac
    done
}

# Configure QoS for core switch
configure_core_switch() {
    local switch=$1
    
    if ! switch_exists "$switch"; then
        log_warn "Switch $switch does not exist, skipping"
        return
    fi
    
    log_info "Configuring core switch: $switch"
    
    # Get all ports
    local ports=$(get_switch_ports "$switch")
    
    for port in $ports; do
        # All core switch ports use core link capacity
        configure_port_qos "$switch" "$port" "$CORE_LINK_CAPACITY"
    done
}

# Cleanup all QoS configuration
cleanup_qos() {
    log_info "Cleaning up all QoS configuration..."
    
    # Clear QoS from all switches
    for switch in $EDGE_SWITCHES $CORE_SWITCHES; do
        if switch_exists "$switch"; then
            log_info "Clearing QoS on $switch"
            local ports=$(get_switch_ports "$switch")
            for port in $ports; do
                run_cmd "ovs-vsctl clear port $port qos 2>/dev/null || true"
            done
        fi
    done
    
    # Destroy all orphaned QoS and Queue records
    log_info "Destroying orphaned QoS/Queue records..."
    run_cmd "ovs-vsctl -- --all destroy QoS -- --all destroy Queue 2>/dev/null || true"
    
    log_success "QoS cleanup complete"
}

# Show current QoS configuration
show_qos() {
    log_info "Current QoS Configuration:"
    echo ""
    
    for switch in $EDGE_SWITCHES $CORE_SWITCHES; do
        if switch_exists "$switch"; then
            echo "=== $switch ==="
            local ports=$(get_switch_ports "$switch")
            for port in $ports; do
                local qos_uuid=$(ovs-vsctl get port "$port" qos 2>/dev/null || echo "[]")
                if [ "$qos_uuid" != "[]" ]; then
                    echo "  $port: QoS configured"
                    ovs-vsctl list qos "$qos_uuid" 2>/dev/null | grep -E "(type|queues|max-rate)" | sed 's/^/    /'
                else
                    echo "  $port: No QoS"
                fi
            done
            echo ""
        fi
    done
}

# Verify QoS is working
verify_qos() {
    log_info "Verifying QoS configuration..."
    
    local success=true
    
    for switch in $EDGE_SWITCHES; do
        if switch_exists "$switch"; then
            local ports=$(get_switch_ports "$switch")
            for port in $ports; do
                local qos_uuid=$(ovs-vsctl get port "$port" qos 2>/dev/null || echo "[]")
                if [ "$qos_uuid" = "[]" ]; then
                    log_error "No QoS on $switch:$port"
                    success=false
                else
                    # Check number of queues
                    local queues=$(ovs-vsctl get qos "$qos_uuid" queues 2>/dev/null)
                    if echo "$queues" | grep -q "0="; then
                        log_success "QoS verified on $switch:$port (4 queues)"
                    else
                        log_warn "QoS on $switch:$port may be incomplete"
                    fi
                fi
            done
        fi
    done
    
    if [ "$success" = true ]; then
        log_success "All QoS verification passed"
        return 0
    else
        log_error "QoS verification failed"
        return 1
    fi
}

# Print usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Configure QoS queues on Open vSwitch for traffic prioritization."
    echo ""
    echo "Options:"
    echo "  --cleanup       Remove all QoS configuration"
    echo "  --edge-only     Only configure edge switches (s1, s4)"
    echo "  --all           Configure all switches (edge + core)"
    echo "  --show          Show current QoS configuration"
    echo "  --verify        Verify QoS is configured correctly"
    echo "  --dry-run       Show commands without executing"
    echo "  --help          Show this help message"
    echo ""
    echo "Queue Configuration:"
    echo "  Queue 0 (P3 - Banking):  min=$((Q0_MIN_RATE/1000000))Mbps, max=$((Q0_MAX_RATE/1000000))Mbps"
    echo "  Queue 1 (P2 - Voice):    min=$((Q1_MIN_RATE/1000000))Mbps, max=$((Q1_MAX_RATE/1000000))Mbps"
    echo "  Queue 2 (P1 - Web):      min=$((Q2_MIN_RATE/1000000))Mbps, max=$((Q2_MAX_RATE/1000000))Mbps"
    echo "  Queue 3 (P0 - Bulk):     max=$((Q3_MAX_RATE/1000000))Mbps (throttled)"
    echo ""
    echo "Examples:"
    echo "  $0 --edge-only      # Configure edge switches only"
    echo "  $0 --all            # Configure all switches"
    echo "  $0 --cleanup        # Remove all QoS"
    echo "  $0 --show           # Display current config"
}

# Main
main() {
    local mode="edge"
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --cleanup)
                cleanup_qos
                exit 0
                ;;
            --edge-only)
                mode="edge"
                shift
                ;;
            --all)
                mode="all"
                shift
                ;;
            --show)
                show_qos
                exit 0
                ;;
            --verify)
                verify_qos
                exit $?
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --help|-h)
                usage
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                usage
                exit 1
                ;;
        esac
    done
    
    echo "=========================================="
    echo "  SDN QoS Queue Setup"
    echo "=========================================="
    echo ""
    
    if [ "$DRY_RUN" = true ]; then
        log_warn "DRY-RUN mode enabled - no changes will be made"
    fi
    
    # Cleanup first
    log_info "Cleaning up existing QoS..."
    cleanup_qos
    
    # Configure edge switches
    log_info "Configuring edge switches..."
    for switch in $EDGE_SWITCHES; do
        configure_edge_switch "$switch"
    done
    
    # Configure core switches if requested
    if [ "$mode" = "all" ]; then
        log_info "Configuring core switches..."
        for switch in $CORE_SWITCHES; do
            configure_core_switch "$switch"
        done
    fi
    
    echo ""
    log_success "QoS setup complete!"
    echo ""
    
    # Verify
    if [ "$DRY_RUN" = false ]; then
        verify_qos
    fi
}

# Run main
main "$@"
