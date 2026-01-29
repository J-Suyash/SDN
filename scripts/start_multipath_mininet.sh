#!/bin/bash
# Start Multi-Path Mininet on Host Machine
# Phase 2: 6-switch topology

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# 1. Check for Mininet
if ! command -v mn &> /dev/null; then
    log_error "Mininet (mn) not found. Please install it: sudo apt install mininet"
    exit 1
fi

# 2. Check for OVS
if ! command -v ovs-vsctl &> /dev/null; then
    log_error "Open vSwitch not found. Please install it: sudo apt install openvswitch-switch"
    exit 1
fi

# 3. Ensure OVS service is running
if ! systemctl is-active --quiet openvswitch-switch; then
    log_info "Starting Open vSwitch service..."
    sudo systemctl start openvswitch-switch
fi

# 4. Clean up previous runs
log_info "Cleaning up Mininet..."
sudo mn -c

# 5. Check if Faucet is reachable in Docker
log_info "Checking if Faucet is reachable on 127.0.0.1:6653..."
if ! nc -z 127.0.0.1 6653; then
    log_warn "Faucet not detected on port 6653. Ensure docker containers are running!"
    log_warn "Run: sudo docker compose -f docker/docker-compose.yml up -d"
fi

# 6. Launch Topology
log_info "Starting Multi-Path Mininet topology..."
log_info "Connecting to Faucet at 127.0.0.1:6653"
sudo python3 topologies/topo_multipath.py
