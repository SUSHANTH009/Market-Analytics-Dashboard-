from abc import ABC, abstractmethod
from typing import Callable
from datetime import datetime


class BaseExchangeConnector(ABC):
    
    def __init__(self, api_key: str = None, api_secret: str = None):
        self.api_key = api_key
        self.api_secret = api_secret
        self.exchange_name = self._get_exchange_name()
    
    @abstractmethod
    def _get_exchange_name(self) -> str:
        pass
    
    @abstractmethod
    async def subscribe_trades(self, symbols: list, callback: Callable):
        pass
    
    @abstractmethod
    async def disconnect(self):
        pass
    
    def normalize_symbol(self, symbol: str) -> str:
        return symbol.upper()
    
    def get_metadata(self) -> dict:
        return {
            "exchange": self.exchange_name,
            "timestamp": datetime.utcnow().isoformat()
        }
