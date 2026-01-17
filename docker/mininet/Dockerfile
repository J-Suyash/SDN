# Mininet + Open vSwitch + Python Environment
# For SDN Traffic Management Project

FROM ubuntu:22.04

# Avoid interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # Mininet and OVS
    mininet \
    openvswitch-switch \
    openvswitch-testcontroller \
    # Network tools
    iperf \
    iperf3 \
    iputils-ping \
    net-tools \
    iproute2 \
    tcpdump \
    tshark \
    curl \
    wget \
    # Python
    python3 \
    python3-pip \
    python3-dev \
    # Build tools
    build-essential \
    git \
    # Utilities
    vim \
    less \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

# Additional Python packages for Mininet
RUN pip3 install --no-cache-dir \
    mininet \
    scapy \
    requests \
    pandas \
    numpy

# Create app directory
WORKDIR /app

# Create necessary directories
RUN mkdir -p /app/mininet /app/collector /app/data

# Start OVS service script
COPY start-ovs.sh /usr/local/bin/start-ovs.sh
RUN chmod +x /usr/local/bin/start-ovs.sh

# Default command
CMD ["/usr/local/bin/start-ovs.sh"]
