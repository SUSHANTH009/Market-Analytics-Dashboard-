import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Callable
from collections import deque


logger = logging.getLogger(__name__)


class AlertManager:
    
    def __init__(self, analytics_engine, config=None):
        self.analytics = analytics_engine
        self.config = config
        self.alerts = []
        self.alert_history = deque(maxlen=100) 
        self.is_running = False
        self._alert_callbacks = []
        logger.info("AlertManager initialized")
    
    def add_alert(self, alert_config: Dict):

        alert_id = alert_config.get('id', f"alert_{len(self.alerts)}")
        alert_config['id'] = alert_id
        alert_config['created_at'] = datetime.utcnow().isoformat()
        
        self.alerts.append(alert_config)
        logger.info(f"✅ Alert added: {alert_id} - {alert_config['type']} {alert_config['condition']} {alert_config['threshold']}")
        print(f"✅ Alert added: {alert_id}")
        return alert_id
    
    def remove_alert(self, alert_id: str):

        before_count = len(self.alerts)
        self.alerts = [a for a in self.alerts if a.get('id') != alert_id]
        after_count = len(self.alerts)
        
        if before_count > after_count:
            logger.info(f"🗑️ Alert removed: {alert_id}")
            print(f"🗑️ Alert removed: {alert_id}")
        else:
            logger.warning(f"Alert not found: {alert_id}")
    
    def get_alerts(self) -> List[Dict]:

        return self.alerts
    
    def get_alert_history(self) -> List[Dict]:

        return list(self.alert_history)
    
    def clear_history(self):

        history_count = len(self.alert_history)
        self.alert_history.clear()
        logger.info(f"Alert history cleared ({history_count} items)")
    
    def register_callback(self, callback: Callable):

        self._alert_callbacks.append(callback)
        logger.info(f"Alert callback registered (total: {len(self._alert_callbacks)})")
    
    def _check_condition(self, value: float, condition: str, threshold: float) -> bool:

        try:
            if condition == '>':
                return value > threshold
            elif condition == '<':
                return value < threshold
            elif condition == '>=':
                return value >= threshold
            elif condition == '<=':
                return value <= threshold
            elif condition == '==':
                return abs(value - threshold) < 0.0001
            else:
                logger.warning(f"Unknown condition: {condition}")
                return False
        except Exception as e:
            logger.error(f"Error checking condition: {e}")
            return False
    
    async def _check_alert(self, alert: Dict) -> bool:

        if not alert.get('enabled', True):
            return False
        
        alert_id = alert.get('id', 'unknown')
        
        try:
            exchange = alert.get('exchange', 'BINANCE')
            symbol1 = alert.get('symbol1')
            symbol2 = alert.get('symbol2')
            alert_type = alert.get('type')
            condition = alert.get('condition', '>')
            threshold = float(alert.get('threshold', 0))
            timeframe = alert.get('timeframe', '1m')
            
            if not symbol1:
                logger.warning(f"Alert {alert_id} missing symbol1")
                return False
            
            logger.debug(f"Checking alert {alert_id}: {symbol1} {alert_type} {condition} {threshold}")
            
            analytics = self.analytics.compute_comprehensive_analytics(
                exchange=exchange,
                symbol1=symbol1,
                symbol2=symbol2,
                timeframe=timeframe,
                limit=100
            )
            
            if alert_type == 'zscore':
                if 'current_zscore' not in analytics:
                    logger.debug(f"Alert {alert_id}: No z-score data available")
                    return False
                value = analytics['current_zscore']
                
            elif alert_type == 'price':
                if 'symbol1_stats' not in analytics:
                    logger.debug(f"Alert {alert_id}: No price stats available")
                    return False
                value = analytics['symbol1_stats'].get('current_price', 0)
                
            elif alert_type == 'correlation':
                if 'current_correlation' not in analytics:
                    logger.debug(f"Alert {alert_id}: No correlation data available")
                    return False
                value = analytics['current_correlation']
                
            elif alert_type == 'spread':
                if 'spread_stats' not in analytics:
                    logger.debug(f"Alert {alert_id}: No spread data available")
                    return False
                value = analytics['spread_stats'].get('current_price', 0)
                
            elif alert_type == 'price_change':
                if 'symbol1_stats' not in analytics:
                    logger.debug(f"Alert {alert_id}: No price change data available")
                    return False
                value = analytics['symbol1_stats'].get('change_pct', 0)
                
            else:
                logger.warning(f"Alert {alert_id}: Unknown alert type: {alert_type}")
                return False
            

            if self._check_condition(value, condition, threshold):

                alert_event = {
                    'alert_id': alert['id'],
                    'triggered_at': datetime.utcnow().isoformat(),
                    'symbol1': symbol1,
                    'symbol2': symbol2,
                    'type': alert_type,
                    'condition': f"{condition} {threshold}",
                    'actual_value': value,
                    'message': f"{alert_type.upper()} alert: {symbol1} {condition} {threshold} (actual: {value:.4f})"
                }
                
                self.alert_history.append(alert_event)
                logger.warning(f"🔔 ALERT TRIGGERED: {alert_event['message']}")
                
                for callback in self._alert_callbacks:
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(alert_event)
                        else:
                            callback(alert_event)
                    except Exception as e:
                        logger.error(f"❌ Error in alert callback: {e}", exc_info=True)
                        print(f"❌ Error in alert callback: {e}")
                
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Error checking alert {alert_id}: {e}", exc_info=True)
            print(f"❌ Error checking alert: {e}")
            return False
    
    async def run(self, check_interval: int = 1):

        self.is_running = True
        logger.info(f"🔔 Alert manager started (checking every {check_interval}s)")
        print(f"🔔 Alert manager started (checking every {check_interval}s)")
        
        check_count = 0
        
        while self.is_running:
            try:
                check_count += 1
                
                if len(self.alerts) == 0:
                    logger.debug(f"Alert check #{check_count}: No alerts configured")
                else:
                    logger.debug(f"Alert check #{check_count}: Checking {len(self.alerts)} alerts")
                    
                    # Check all enabled alerts
                    for alert in self.alerts:
                        if alert.get('enabled', True):
                            await self._check_alert(alert)
                
                await asyncio.sleep(check_interval)
                
            except asyncio.CancelledError:
                logger.info("Alert manager cancelled")
                break
            except Exception as e:
                logger.error(f"❌ Error in alert loop: {e}", exc_info=True)
                print(f"❌ Error in alert loop: {e}")
                await asyncio.sleep(check_interval)
        
        logger.info("Alert manager stopped")
    
    def stop(self):
        """Stop alert checking"""
        self.is_running = False
        logger.info("🛑 Alert manager stopped")
        print("🛑 Alert manager stopped")
