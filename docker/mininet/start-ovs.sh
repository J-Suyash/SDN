#!/bin/bash
# Start Open vSwitch service and keep container running

echo "Starting Open vSwitch..."

# Create OVS directories if they don't exist
mkdir -p /var/run/openvswitch
mkdir -p /var/log/openvswitch
mkdir -p /etc/openvswitch

# Check if OVS database exists, create if not
if [ ! -f /etc/openvswitch/conf.db ]; then
    echo "Creating OVS database..."
    ovsdb-tool create /etc/openvswitch/conf.db /usr/share/openvswitch/vswitch.ovsschema
fi

# Start ovsdb-server
ovsdb-server --remote=punix:/var/run/openvswitch/db.sock \
    --remote=db:Open_vSwitch,Open_vSwitch,manager_options \
    --private-key=db:Open_vSwitch,SSL,private_key \
    --certificate=db:Open_vSwitch,SSL,certificate \
    --bootstrap-ca-cert=db:Open_vSwitch,SSL,ca_cert \
    --pidfile --detach --log-file

# Initialize OVS database
ovs-vsctl --no-wait init

# Start ovs-vswitchd
ovs-vswitchd --pidfile --detach --log-file

echo "Open vSwitch started successfully"
echo "OVS version: $(ovs-vsctl --version | head -1)"

# Keep container running
exec tail -f /dev/null
