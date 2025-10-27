"""Faucet SDN controller integration"""

import asyncio
import yaml
from typing import Dict, List, Any, Optional
import aiohttp


class FaucetController:
    """Wrapper for Faucet SDN controller"""
    
    def __init__(self, faucet_api_url: str = "http://localhost:9302"):
        self.api_url = faucet_api_url
        self.config = {}
    
    async def configure_switch(self, config: Dict[str, Any]):
        """Configure Faucet switch"""
        self.config = config
        
        faucet_yaml = self._generate_faucet_yaml(config)
        
        config_file = '/etc/faucet/faucet.yaml'
        with open(config_file, 'w') as f:
            yaml.dump(faucet_yaml, f)
    
    def _generate_faucet_yaml(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate Faucet YAML configuration"""
        return {
            'vlans': config.get('vlans', {}),
            'dps': config.get('dps', {}),
            'routers': config.get('routers', {}),
        }
    
    async def install_flow_rule(self, rule: Dict[str, Any]) -> bool:
        """Install flow rule via Faucet API"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.api_url}/api/v1/install_flow",
                    json=rule
                ) as response:
                    return response.status == 200
        except:
            return False
    
    async def get_switch_stats(self, dpid: str) -> Dict[str, Any]:
        """Get switch statistics"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.api_url}/api/v1/switches/{dpid}/stats"
                ) as response:
                    if response.status == 200:
                        return await response.json()
        except:
            pass
        
        return {}
    
    async def update_qos(self, dpid: str, port: str, qos_config: Dict[str, Any]) -> bool:
        """Update QoS configuration"""
        rule = {
            'dpid': dpid,
            'port': port,
            'qos': qos_config
        }
        return await self.install_flow_rule(rule)
    
    async def reroute_flow(self, flow_id: str, new_path: List[str]) -> bool:
        """Reroute flow to new path"""
        rule = {
            'flow_id': flow_id,
            'path': new_path,
            'action': 'reroute'
        }
        return await self.install_flow_rule(rule)
    
    async def rate_limit(self, source: str, max_rate: float) -> bool:
        """Apply rate limiting"""
        rule = {
            'match': {'ip_src': source},
            'actions': {'rate_limit': max_rate}
        }
        return await self.install_flow_rule(rule)
