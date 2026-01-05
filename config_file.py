import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class RedisConfig:

    host: str = os.getenv("REDIS_HOST", "localhost")
    port: int = int(os.getenv("REDIS_PORT", "6379"))
    db: int = int(os.getenv("REDIS_DB", "0"))
    password: str = os.getenv("REDIS_PASSWORD", None)
    decode_responses: bool = True


@dataclass
class BinanceConfig:

    api_key: str = os.getenv("BINANCE_API_KEY", None)
    api_secret: str = os.getenv("BINANCE_API_SECRET", None)


@dataclass
class DataConfig:
    default_symbols: List[str] = field(default_factory=lambda: ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT"])
    
    timeframes: List[str] = field(default_factory=lambda: ["1s", "1m", "5m", "15m", "1h"])
    
    max_trades_per_key: int = 510000
    trades_to_delete: int = 10000


@dataclass
class AnalyticsConfig:
    default_lookback: int = 500  # Default data points for analysis
    z_score_window: int = 20
    correlation_window: int = 30
    adf_max_lag: int = 10
    
    # not used currently, but reserved for future use
    rolling_windows: List[int] = field(default_factory=lambda: [10, 20, 30, 50, 100])
    
    regression_types: List[str] = field(default_factory=lambda: ["OLS", "Rolling OLS"])


@dataclass
class DashboardConfig:
    """Dashboard configuration"""
    host: str = "0.0.0.0"
    port: int = 8050
    debug: bool = True
    
    #not used currently, but reserved for future use
    fast_update_interval: int = 500  
    medium_update_interval: int = 1000  
    slow_update_interval: int = 5000 


@dataclass
class AlertConfig:
    check_interval: int = 1  
    default_alerts: List[dict] = field(default_factory=lambda: [
        {"type": "zscore", "condition": ">", "threshold": 2.0, "enabled": True},
        {"type": "zscore", "condition": "<", "threshold": -2.0, "enabled": True},
    ])


redis_config = RedisConfig()
binance_config = BinanceConfig()
data_config = DataConfig()
analytics_config = AnalyticsConfig()
dashboard_config = DashboardConfig()
alert_config = AlertConfig()


def get_redis_config() -> RedisConfig:
    return redis_config


def get_binance_config() -> BinanceConfig:
    return binance_config


def get_data_config() -> DataConfig:
    return data_config


def get_analytics_config() -> AnalyticsConfig:
    return analytics_config


def get_dashboard_config() -> DashboardConfig:
    return dashboard_config


def get_alert_config() -> AlertConfig:
    return alert_config


def print_config():
    """Print current configuration"""
    print("\n" + "="*70)
    print("🔧 CONFIGURATION")
    print("="*70)
    
    print(f"\n[Redis]")
    print(f"  Host: {redis_config.host}:{redis_config.port}")
    print(f"  Database: {redis_config.db}")
    print(f"  Max Trades Per Key: {data_config.max_trades_per_key:,}")
    print(f"  Delete When Full: {data_config.trades_to_delete:,} trades")
    
    print(f"\n[Data Collection]")
    print(f"  Symbols: {', '.join(data_config.default_symbols)}")
    print(f"  Timeframes: {', '.join(data_config.timeframes)}")
    
    print(f"\n[Analytics]")
    print(f"  Lookback: {analytics_config.default_lookback} points")
    print(f"  Z-Score Window: {analytics_config.z_score_window}")
    
    print(f"\n[Dashboard]")
    print(f"  URL: http://{dashboard_config.host}:{dashboard_config.port}")
    print(f"  Fast Update: {dashboard_config.fast_update_interval}ms")
    
    print(f"\n[Alerts]")
    print(f"  Check Interval: {alert_config.check_interval}s")
    print(f"  Default Alerts: {len(alert_config.default_alerts)} configured")
    
    print("="*70 + "\n")
