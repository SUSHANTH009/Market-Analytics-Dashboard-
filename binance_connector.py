import json
import asyncio
import websockets
import logging
from datetime import datetime
from typing import Callable, List
from base_connector import BaseExchangeConnector

logger = logging.getLogger(__name__)


class BinanceConnector(BaseExchangeConnector):
    
    WS_BASE_URL = "wss://fstream.binance.com/stream?streams="
    
    def __init__(self, api_key=None, api_secret=None):
        super().__init__(api_key, api_secret)
        self._websocket = None
        self._running = False
        logger.info(f"BinanceConnector initialized for exchange: {self.exchange_name}")
    
    def _get_exchange_name(self) -> str:
        return "BINANCE"
    
    def _normalize_trade(self, data: dict) -> dict:
        try:
            timestamp_ms = data.get('T') or data.get('E')
            ts_iso = datetime.utcfromtimestamp(timestamp_ms / 1000.0).isoformat() + "Z"
            
            normalized = {
                "exchange": self.exchange_name,
                "symbol": data.get('s'),
                "ts": ts_iso,
                "timestamp_ms": timestamp_ms,
                "price": float(data.get('p')),
                "size": float(data.get('q')),
                "trade_id": data.get('t'),
                "is_buyer_maker": data.get('m', False)
            }
            
            return normalized
            
        except Exception as e:
            logger.error(f"Error normalizing trade data: {e}", exc_info=True)
            raise
    
    async def subscribe_trades(self, symbols: List[str], callback: Callable):

        streams = "/".join([f"{s.lower()}@trade" for s in symbols])
        url = f"{self.WS_BASE_URL}{streams}"
        
        logger.info(f"🔗 Connecting to Binance WebSocket...")
        logger.info(f"📊 Symbols: {', '.join(symbols)}")
        logger.info(f"🌐 URL: {url}")
        
        print(f"🔗 Connecting to Binance WebSocket...")
        print(f"📊 Symbols: {', '.join(symbols)}")
        
        self._running = True
        connection_attempt = 0
        
        while self._running:
            connection_attempt += 1
            logger.info(f"Connection attempt #{connection_attempt}")
            
            try:
                async with websockets.connect(url) as websocket:
                    self._websocket = websocket
                    logger.info("✅ WebSocket connected. Streaming trades...")
                    print("✅ WebSocket connected. Streaming trades...")
                    
                    message_count = 0
                    
                    while self._running:
                        try:
                            message = await websocket.recv()
                            message_count += 1
                            
                            if message_count % 100 == 0:
                                logger.debug(f"Received {message_count} messages")
                            
                            raw_json = json.loads(message)
                            trade_data = raw_json.get('data', {})
                            
                            if trade_data.get('e') == 'trade':
                                normalized_data = self._normalize_trade(trade_data)
                                
                                if asyncio.iscoroutinefunction(callback):
                                    await callback(normalized_data)
                                else:
                                    callback(normalized_data)
                                    
                        except websockets.ConnectionClosed as e:
                            logger.warning(f"⚠️ WebSocket connection closed: {e}")
                            print("\n⚠️ WebSocket connection closed by server.")
                            
                            if self._running:
                                logger.info("🔄 Attempting to reconnect in 5 seconds...")
                                print("🔄 Attempting to reconnect...")
                                await asyncio.sleep(5)
                            break
                            
                        except json.JSONDecodeError as e:
                            logger.error(f"Error decoding JSON message: {e}")
                            continue
                            
                        except Exception as e:
                            logger.error(f"\n❌ Error processing message: {e}", exc_info=True)
                            print(f"\n❌ Error processing message: {e}")
                            await asyncio.sleep(1)
                
            except websockets.exceptions.WebSocketException as e:
                logger.error(f"\n❌ WebSocket connection error: {e}")
                print(f"\n❌ WebSocket connection error: {e}")
                
                if self._running:
                    logger.info("🔄 Reconnecting in 5 seconds...")
                    print("🔄 Reconnecting in 5 seconds...")
                    await asyncio.sleep(5)
                    
            except Exception as e:
                logger.error(f"\n❌ Unexpected error in WebSocket: {e}", exc_info=True)
                print(f"\n❌ Unexpected error: {e}")
                
                if self._running:
                    await asyncio.sleep(5)
        
        logger.info("WebSocket subscription loop ended")
    
    async def disconnect(self):
        logger.info("Disconnecting WebSocket...")
        self._running = False
        
        if self._websocket:
            try:
                await self._websocket.close()
                logger.info("✅ WebSocket closed successfully")
            except Exception as e:
                logger.error(f"Error closing WebSocket: {e}")
        
        print("🛑 WebSocket disconnected")
