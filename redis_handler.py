import json
import redis
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class RedisHandler:

    TIMEFRAME_MAP = {
        'tick': 'tick',
        '1s': '1S',
        '5s': '5S',
        '10s': '10S',
        '30s': '30S',
        '1m': '1T',
        '5m': '5T',
        '15m': '15T',
        '30m': '30T',
        '1h': '1H',
        '4h': '4H',
        '1d': '1D',
    }
    
    MAX_TRADES_PER_KEY = 510000
    TRADES_TO_DELETE = 10000
    
    def __init__(self, host: str = "localhost", port: int = 6379, 
                 db: int = 0, password: Optional[str] = None, 
                 decode_responses: bool = True):
        self.redis_client = redis.Redis(
            host=host,
            port=port,
            db=db,
            password=password,
            decode_responses=decode_responses
        )
        self._test_connection()
        self.clear_all_trade_data()
        logger.info(f"RedisHandler initialized with max trades per key: {self.MAX_TRADES_PER_KEY}")
    
    def _normalize_timeframe(self, timeframe: str) -> str:
        timeframe_lower = timeframe.lower().strip()
        
        if timeframe_lower in self.TIMEFRAME_MAP:
            result = self.TIMEFRAME_MAP[timeframe_lower]
            logger.debug(f"Normalized timeframe '{timeframe}' -> '{result}'")
            return result
        
        logger.warning(f"Unknown timeframe '{timeframe}', using as-is. This may cause issues.")
        return timeframe
    
    def _test_connection(self):
        try:
            self.redis_client.ping()
            logger.info("✅ Successfully connected to Redis")
            print("✅ Successfully connected to Redis")
        except redis.ConnectionError as e:
            logger.error(f"❌ Failed to connect to Redis: {e}")
            print(f"❌ Failed to connect to Redis: {e}")
            raise
    
    def clear_all_trade_data(self):
        try:
            # Find all trade-related keys
            pattern = "trades:raw:*"
            keys = self.redis_client.keys(pattern)
            
            if keys:
                deleted_count = self.redis_client.delete(*keys)
                logger.info(f"🗑️ Cleared {deleted_count} trade data keys from Redis")
                print(f"🗑️ Cleared {deleted_count} trade data keys from previous session")
            else:
                logger.info("✅ No previous trade data found in Redis")
                print("✅ Redis is clean, no previous trade data found")
                
        except Exception as e:
            logger.error(f"❌ Error clearing Redis data: {e}")
            print(f"❌ Error clearing Redis data: {e}")
            raise
    
    def store_trade(self, trade_data: Dict):
        if not trade_data:
            return False
        
        try:
            exchange = trade_data.get("exchange", "UNKNOWN")
            symbol = trade_data.get("symbol", "UNKNOWN")
            timestamp_ms = trade_data.get("timestamp_ms")
            
            key = f"trades:raw:{exchange}:{symbol}"
            self.redis_client.zadd(key, {json.dumps(trade_data): timestamp_ms})
            

            current_size = self.redis_client.zcard(key)
            
            if current_size > self.MAX_TRADES_PER_KEY:

                deleted_count = self.redis_client.zremrangebyrank(key, 0, self.TRADES_TO_DELETE - 1)
                logger.info(f"Trimmed {deleted_count} trades from {key} (size was {current_size})")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error storing trade: {e}", exc_info=True)
            return False
    
    def get_trades_raw(self, exchange: str, symbol: str, limit: int = 1000) -> List[Dict]:
        try:
            key = f"trades:raw:{exchange}:{symbol}"
            data = self.redis_client.zrevrange(key, 0, limit - 1)
            trades = [json.loads(item) for item in data]
            logger.debug(f"Retrieved {len(trades)} raw trades for {exchange}:{symbol}")
            return trades
        except Exception as e:
            logger.error(f"❌ Error retrieving trades for {exchange}:{symbol}: {e}")
            return []
    
    # def get_latest_trade(self, exchange: str, symbol: str) -> Optional[Dict]:
    #     """Get the latest trade for a symbol"""
    #     try:
    #         key = f"trades:raw:{exchange}:{symbol}"
    #         data = self.redis_client.zrevrange(key, 0, 0)
    #         if data:
    #             return json.loads(data[0])
    #         return None
    #     except Exception as e:
    #         logger.error(f"❌ Error retrieving latest trade for {exchange}:{symbol}: {e}")
    #         return None
    
    def resample_trades(self, exchange: str, symbol: str, timeframe: str, limit: int = 1000) -> pd.DataFrame:
        try:
            trades = self.get_trades_raw(exchange, symbol, limit=limit)
            
            if not trades:
                logger.warning(f"No trades found for {exchange}:{symbol}")
                return pd.DataFrame()
            
            logger.info(f"Resampling {len(trades)} trades for {exchange}:{symbol} to {timeframe}")
            
            df = pd.DataFrame(trades)
            
            if 'price' not in df.columns or 'size' not in df.columns:
                logger.error(f"Missing required columns in trade data for {exchange}:{symbol}")
                return pd.DataFrame()
            
            # Parse timestamps properly - handle both ISO8601 and Unix timestamps
            try:
                # Try parsing as ISO8601 first
                if 'ts' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['ts'], utc=True, errors='coerce', infer_datetime_format=True)
                elif 'timestamp_ms' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp_ms'], unit='ms', utc=True)
                else:
                    logger.error(f"No timestamp column found in trade data")
                    return pd.DataFrame()
            except Exception as e:
                logger.error(f"Error parsing timestamps: {e}")
                # Fallback to millisecond Unix timestamp
                if 'timestamp_ms' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp_ms'], unit='ms', utc=True)
                else:
                    logger.error(f"Cannot parse timestamps for {exchange}:{symbol}")
                    return pd.DataFrame()
            
            df = df.set_index('timestamp')
            df = df.sort_index()
            
            df['price'] = pd.to_numeric(df['price'], errors='coerce')
            df['size'] = pd.to_numeric(df['size'], errors='coerce')
            
            df = df.dropna(subset=['price', 'size'])
            
            if len(df) == 0:
                logger.warning(f"No valid data after cleaning for {exchange}:{symbol}")
                return pd.DataFrame()
            
            logger.debug(f"DataFrame created with {len(df)} rows, index type: {type(df.index)}, dtype: {df.index.dtype}")
            
            pandas_timeframe = self._normalize_timeframe(timeframe)
            
            logger.debug(f"Using pandas timeframe: {pandas_timeframe}")
            
            ohlcv = pd.DataFrame()
            
            ohlcv['open'] = df['price'].resample(pandas_timeframe, label='right', closed='right').first()
            ohlcv['high'] = df['price'].resample(pandas_timeframe, label='right', closed='right').max()
            ohlcv['low'] = df['price'].resample(pandas_timeframe, label='right', closed='right').min()
            ohlcv['close'] = df['price'].resample(pandas_timeframe, label='right', closed='right').last()
            ohlcv['volume'] = df['size'].resample(pandas_timeframe, label='right', closed='right').sum()
            ohlcv['trades'] = df['size'].resample(pandas_timeframe, label='right', closed='right').count()
            
            # Remove incomplete candles - only keep candles where next period has data
            if len(ohlcv) > 1:
                # Mark all candles as complete except the last one
                ohlcv = ohlcv.iloc[:-1]
                logger.debug(f"Keeping {len(ohlcv)} complete candles (excluding last incomplete candle)")
            else:
                # If only one candle, it's incomplete
                logger.debug("Only one candle available, which is incomplete. Returning empty DataFrame.")
                return pd.DataFrame()
            

            ohlcv = ohlcv.dropna(subset=['close'])
            

            if len(ohlcv) > 0:
                ohlcv['open'] = ohlcv['open'].fillna(method='ffill')
                ohlcv['high'] = ohlcv['high'].fillna(ohlcv['close'])
                ohlcv['low'] = ohlcv['low'].fillna(ohlcv['close'])
                ohlcv['volume'] = ohlcv['volume'].fillna(0)
                ohlcv['trades'] = ohlcv['trades'].fillna(0)
            
            result = ohlcv.tail(limit)
            logger.info(f"✅ Resampled to {len(result)} complete {timeframe} candles (pandas: {pandas_timeframe})")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error resampling trades for {exchange}:{symbol}:{timeframe}: {e}", exc_info=True)
            import traceback
            traceback.print_exc()
            return pd.DataFrame()
    
    def get_price_series(self, exchange: str, symbol: str, timeframe: str = "tick", limit: int = 1000) -> pd.Series:
        try:
            if timeframe == "tick":
                trades = self.get_trades_raw(exchange, symbol, limit)
                
                if not trades:
                    logger.warning(f"No tick data for {exchange}:{symbol}")
                    return pd.Series(dtype=float)
                
                df = pd.DataFrame(trades)
                
                try:
                    if 'ts' in df.columns:
                        df['timestamp'] = pd.to_datetime(df['ts'], utc=True, errors='coerce', infer_datetime_format=True)
                    else:
                        df['timestamp'] = pd.to_datetime(df['timestamp_ms'], unit='ms', utc=True)
                except:
                    df['timestamp'] = pd.to_datetime(df['timestamp_ms'], unit='ms', utc=True)
                
                df['price'] = pd.to_numeric(df['price'], errors='coerce')
                df = df.dropna(subset=['price', 'timestamp'])
                
                result = pd.Series(df['price'].values, index=df['timestamp'])
                result = result.sort_index()
                logger.debug(f"Retrieved {len(result)} tick prices for {exchange}:{symbol}")
                return result
            else:
                df = self.resample_trades(exchange, symbol, timeframe, limit)
                
                if not df.empty and 'close' in df.columns:
                    logger.debug(f"Retrieved {len(df)} {timeframe} close prices for {exchange}:{symbol}")
                    return df['close']
                else:
                    logger.warning(f"No price data for {exchange}:{symbol}:{timeframe}")
                    return pd.Series(dtype=float)
                
        except Exception as e:
            logger.error(f"❌ Error getting price series for {exchange}:{symbol}:{timeframe}: {e}", exc_info=True)
            return pd.Series(dtype=float)
    
    def get_volume_series(self, exchange: str, symbol: str, timeframe: str = "1m", 
                         limit: int = 1000) -> pd.Series:
        try:
            df = self.resample_trades(exchange, symbol, timeframe, limit)
            
            if not df.empty and 'volume' in df.columns:
                logger.debug(f"Retrieved {len(df)} volume data points for {exchange}:{symbol}:{timeframe}")
                return df['volume']
            else:
                logger.warning(f"No volume data for {exchange}:{symbol}:{timeframe}")
                return pd.Series(dtype=float)
            
        except Exception as e:
            logger.error(f"❌ Error getting volume series for {exchange}:{symbol}:{timeframe}: {e}", exc_info=True)
            return pd.Series(dtype=float)
    
    # def get_all_symbols(self, exchange: str) -> List[str]:
    #     """Get all symbols currently tracked"""
    #     try:
    #         pattern = f"trades:raw:{exchange}:*"
    #         keys = self.redis_client.keys(pattern)
    #         symbols = [key.split(":")[-1] for key in keys]
    #         unique_symbols = list(set(symbols))
    #         logger.info(f"Found {len(unique_symbols)} symbols for {exchange}")
    #         return unique_symbols
        
    #     except Exception as e:
    #         logger.error(f"❌ Error retrieving symbols for {exchange}: {e}")
    #         return []
    
    def get_stats(self) -> Dict:
        try:
            info = self.redis_client.info()
            stats = {
                "connected_clients": info.get("connected_clients"),
                "used_memory_human": info.get("used_memory_human"),
                "total_keys": self.redis_client.dbsize(),
                "uptime_days": info.get("uptime_in_days")
            }
            logger.debug(f"Redis stats: {stats}")
            return stats
        
        except Exception as e:
            logger.error(f"❌ Error getting Redis stats: {e}")
            return {}
