"""
Base service class for all trading agent services
"""
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class BaseService:
    """Base class for all services in the trading agent"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logger
        
    def validate_config(self, required_keys: list) -> bool:
        """Validate that required configuration keys are present"""
        missing_keys = [key for key in required_keys if key not in self.config]
        if missing_keys:
            self.logger.error(f"Missing required configuration keys: {missing_keys}")
            return False
        return True
    
    def get_config(self, key: str, default: Any = None) -> Any:
        """Get configuration value with default"""
        return self.config.get(key, default)

