# Installation Guide

## Prerequisites

- Python 3.10+
- Ubuntu 20.04 or later
- sudo access

## System Dependencies

```bash
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv
sudo apt-get install -y mininet openvswitch-switch
sudo apt-get install -y prometheus prometheus-node-exporter
```

## Faucet Installation

```bash
git clone https://github.com/faucetsdn/faucet.git
cd faucet
sudo pip3 install -r requirements.txt
sudo python3 setup.py install
cd ..
```

## Project Setup

1. Clone this repository:
```bash
git clone <repository-url>
cd SDN
```

2. Install dependencies using uv:
```bash
uv sync
```

3. Generate training data:
```bash
python scripts/generate_data.py
```

4. Configure systemd services:
```bash
sudo cp config/faucet.yaml /etc/faucet/
sudo cp config/gauge.yaml /etc/faucet/
sudo systemctl enable faucet
sudo systemctl enable gauge
sudo systemctl enable prometheus
```

## Running Scenarios

1. Start system services:
```bash
python scripts/setup_network.py
```

2. Run scenarios:
```bash
python scripts/run_scenario.py --scenario elephant
python scripts/run_scenario.py --scenario multipath
python scripts/run_scenario.py --scenario ddos
python scripts/run_scenario.py --scenario all
```

## Development

Run tests:
```bash
pytest tests/
```

Format code:
```bash
black sdn_ml/
```

Type checking:
```bash
mypy sdn_ml/
```
