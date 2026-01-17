# Faucet SDN Controller
# Official Faucet image with configuration

FROM faucet/faucet:latest

# Faucet configuration is mounted at runtime
# from ../faucet/faucet.yaml

# Expose ports
# 6653 - OpenFlow controller
# 9302 - Prometheus metrics

EXPOSE 6653 9302

# Use default entrypoint from base image
