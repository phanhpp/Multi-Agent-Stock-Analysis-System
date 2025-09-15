"""
Load and manage agent configuration from YAML file
"""

import yaml
from pathlib import Path
from typing import Dict, Any

class AgentConfig:
    """Load and manage agent configuration from YAML file"""
    
    def __init__(self, config_path: str = "config/agent_config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    @property
    def valuation(self) -> Dict[str, float]:
        """Get valuation agent thresholds"""
        return self.config['valuation_agent']
    
    @property
    def sentiment(self) -> Dict[str, float]:
        """Get sentiment agent thresholds"""
        return self.config['sentiment_agent']
    
    @property
    def fundamental(self) -> Dict[str, float]:
        """Get fundamental agent thresholds"""
        return self.config['fundamental_agent']
    
    @property
    def coordinator(self) -> Dict[str, Any]:
        """Get coordinator settings"""
        return self.config['coordinator']

# Global config instance
config = AgentConfig()