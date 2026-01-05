import numpy as np
import pandas as pd
from typing import Dict, Optional
from scipy import stats
from statsmodels.tsa.stattools import adfuller
from statsmodels.regression.linear_model import OLS
from statsmodels.regression.rolling import RollingOLS   
from statsmodels.tools import add_constant
from datetime import datetime
import warnings
import logging

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class AnalyticsEngine:
    """
    Analytics engine for real-time market data analysis
    """
    
    def __init__(self, redis_handler, config=None):
        self.redis = redis_handler
        self.config = config
        logger.info("AnalyticsEngine initialized")
    
    def compute_price_stats(self, prices: pd.Series) -> Dict:
        """Compute comprehensive price statistics"""
        if len(prices) == 0:
            logger.warning("Empty price series provided to compute_price_stats")
            return {}
        
        try:
            # Ensure prices are numeric
            prices = pd.to_numeric(prices, errors='coerce').dropna()
            
            if len(prices) == 0:
                logger.warning("No valid numeric prices after conversion")
                return {}
            
            returns = prices.pct_change().dropna()
            
            stats_dict = {
                'current_price': float(prices.iloc[-1]),
                'mean': float(prices.mean()),
                'median': float(prices.median()),
                'std': float(prices.std()),
                'min': float(prices.min()),
                'max': float(prices.max()),
                'range': float(prices.max() - prices.min()),
                'skewness': float(returns.skew()) if len(returns) > 0 else 0,
                'kurtosis': float(returns.kurtosis()) if len(returns) > 0 else 0,
                'return_mean': float(returns.mean()) if len(returns) > 0 else 0,
                'return_std': float(returns.std()) if len(returns) > 0 else 0,
                'sharpe_ratio': float(returns.mean() / returns.std() * np.sqrt(252)) if len(returns) > 0 and returns.std() != 0 else 0,
                'change_pct': float((prices.iloc[-1] - prices.iloc[0]) / prices.iloc[0] * 100) if len(prices) > 0 and prices.iloc[0] != 0 else 0,
                'data_points': len(prices),
                'timestamp': datetime.utcnow().isoformat()
            }
            
            logger.debug(f"Computed price stats: current_price={stats_dict['current_price']:.2f}, std={stats_dict['std']:.2f}")
            return stats_dict
            
        except Exception as e:
            logger.error(f"❌ Error computing price stats: {e}", exc_info=True)
            return {}
    
    def compute_ols_hedge_ratio(self, y_prices: pd.Series, x_prices: pd.Series, 
                               regression_type: str = "OLS") -> Dict:
        """
        Compute hedge ratio using OLS or Rolling OLS regression
        """
        if len(y_prices) == 0 or len(x_prices) == 0:
            logger.warning("Empty price series provided to compute_ols_hedge_ratio")
            return {}
        
        try:
            # Ensure both series are numeric
            y_prices = pd.to_numeric(y_prices, errors='coerce')
            x_prices = pd.to_numeric(x_prices, errors='coerce')
            
            # Align series by index (timestamp)
            df = pd.DataFrame({'y': y_prices, 'x': x_prices})
            df = df.dropna()
            
            if len(df) < 2:
                logger.warning(f"Insufficient data points ({len(df)}) for regression")
                return {}
            
            logger.info(f"Computing {regression_type} regression with {len(df)} data points")
            
            if regression_type == "Rolling OLS":
                # Rolling OLS with window
                window = min(50, len(df) // 2)
                if window < 2:
                    window = 2
                    
                X = add_constant(df['x'])
                model = RollingOLS(df['y'], X, window=window).fit()
                
                # Get latest parameters - handle both 'const' and 0 index
                params_df = model.params.iloc[-1]
                if 'const' in params_df.index:
                    alpha = float(params_df['const'])
                elif 0 in params_df.index:
                    alpha = float(params_df[0])
                else:
                    alpha = float(params_df.iloc[0])
                
                if 'x' in params_df.index:
                    hedge_ratio = float(params_df['x'])
                elif 1 in params_df.index:
                    hedge_ratio = float(params_df[1])
                else:
                    hedge_ratio = float(params_df.iloc[1])
                
                # Calculate spread using latest hedge ratio
                spread = df['y'] - hedge_ratio * df['x']
                fitted = alpha + hedge_ratio * df['x']
                r_squared = float(model.rsquared.iloc[-1])
                
            else:  # Standard OLS
                X = add_constant(df['x'])
                model = OLS(df['y'], X).fit()
                
                # Handle different parameter indexing - try multiple approaches
                try:
                    # Try named access first
                    hedge_ratio = float(model.params['x'])
                    alpha = float(model.params['const'])
                except (KeyError, TypeError):
                    try:
                        # Try positional access
                        alpha = float(model.params.iloc[0])
                        hedge_ratio = float(model.params.iloc[1])
                    except:
                        # Try integer index
                        alpha = float(model.params[0])
                        hedge_ratio = float(model.params[1])
                
                spread = model.resid
                fitted = model.fittedvalues
                r_squared = float(model.rsquared)
            
            # Convert timestamps properly - handle different index types
            try:
                timestamps_list = []
                for ts in df.tail(500).index:
                    if isinstance(ts, pd.Timestamp):
                        timestamps_list.append(ts.isoformat())
                    elif isinstance(ts, (int, float)):
                        timestamps_list.append(pd.Timestamp(ts, unit='ms').isoformat())
                    else:
                        timestamps_list.append(str(ts))
            except Exception as e:
                logger.warning(f"Error converting timestamps: {e}")
                timestamps_list = [str(i) for i in range(len(df.tail(500)))]
            
            result = {
                'hedge_ratio': hedge_ratio,
                'alpha': alpha,
                'r_squared': r_squared,
                'spread_mean': float(spread.mean()),
                'spread_std': float(spread.std()),
                'residuals': [float(x) if not pd.isna(x) else 0 for x in spread.tail(500).tolist()],
                'fitted_values': [float(x) if not pd.isna(x) else 0 for x in fitted.tail(500).tolist()],
                'timestamps': timestamps_list,
                'regression_type': regression_type,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            logger.info(f"Regression complete: hedge_ratio={hedge_ratio:.4f}, R²={r_squared:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error computing hedge ratio: {e}", exc_info=True)
            import traceback
            traceback.print_exc()
            return {}
    
    def compute_spread(self, price1: pd.Series, price2: pd.Series, 
                      hedge_ratio: float = 1.0, alpha: float = 0.0) -> pd.Series:
        """Compute spread between two price series"""
        try:
            # Ensure numeric
            price1 = pd.to_numeric(price1, errors='coerce')
            price2 = pd.to_numeric(price2, errors='coerce')
            
            df = pd.DataFrame({'p1': price1, 'p2': price2})
            df = df.dropna()
            
            spread = (df['p1'] - alpha) - hedge_ratio * df['p2']
            logger.debug(f"Computed spread with hedge_ratio={hedge_ratio:.4f}, mean={spread.mean():.4f}")
            return spread
        except Exception as e:
            logger.error(f"❌ Error computing spread: {e}", exc_info=True)
            return pd.Series(dtype=float)
    
    def compute_z_score(self, series: pd.Series, window: int = 20) -> pd.Series:
        """Compute rolling z-score"""
        try:
            # Ensure numeric
            series = pd.to_numeric(series, errors='coerce').dropna()
            
            if len(series) < window:
                logger.warning(f"Insufficient data ({len(series)}) for z-score with window={window}")
                return pd.Series(dtype=float)
            
            rolling_mean = series.rolling(window=window, min_periods=1).mean()
            rolling_std = series.rolling(window=window, min_periods=1).std()
            
            # Avoid division by zero
            rolling_std = rolling_std.replace(0, np.nan)
            
            z_score = (series - rolling_mean) / rolling_std
            current_z = z_score.iloc[-1] if len(z_score) > 0 and not pd.isna(z_score.iloc[-1]) else 0
            logger.debug(f"Computed z-score with window={window}, current={current_z:.2f}")
            return z_score
        except Exception as e:
            logger.error(f"❌ Error computing z-score: {e}", exc_info=True)
            return pd.Series(dtype=float)
    
    def perform_adf_test(self, series: pd.Series, max_lag: int = 10) -> Dict:
        """Perform Augmented Dickey-Fuller test"""
        # Ensure numeric and drop NaN
        series = pd.to_numeric(series, errors='coerce').dropna()
        
        # Calculate minimum observations needed for given max_lag
        # Formula: maxlag < (nobs/2 - 1 - ntrend) where ntrend=1
        min_obs_needed = 2 * (max_lag + 2)  # Simplified: 2 * (max_lag + 1 + 1)
        
        if len(series) < min_obs_needed:
            logger.warning(f"Insufficient data for ADF test: {len(series)} < {min_obs_needed} (for maxlag={max_lag})")
            return {}
        
        # Dynamically adjust max_lag if series is not long enough
        # maxlag < (nobs/2 - 2) for safety
        max_possible_lag = int(len(series) / 2) - 2
        adjusted_max_lag = min(max_lag, max_possible_lag)
        
        if adjusted_max_lag < 1:
            logger.warning(f"Series too short for meaningful ADF test: {len(series)} observations")
            return {}
        
        if adjusted_max_lag < max_lag:
            logger.info(f"Adjusted max_lag from {max_lag} to {adjusted_max_lag} due to series length ({len(series)} obs)")
        
        try:
            logger.info(f"Performing ADF test with {len(series)} observations, max_lag={adjusted_max_lag}")
            result = adfuller(series, maxlag=adjusted_max_lag, autolag='AIC')
            
            is_stationary_5 = result[1] < 0.05
            is_stationary_1 = result[1] < 0.01
            
            interpretation = ""
            if result[1] < 0.01:
                interpretation = "✅ Strong evidence of stationarity (p < 0.01)"
            elif result[1] < 0.05:
                interpretation = "✅ Evidence of stationarity (p < 0.05)"
            elif result[1] < 0.10:
                interpretation = "⚠️ Weak evidence of stationarity (p < 0.10)"
            else:
                interpretation = "❌ Not stationary (cannot reject null)"
            
            adf_result = {
                'adf_statistic': float(result[0]),
                'p_value': float(result[1]),
                'used_lag': int(result[2]),
                'n_observations': int(result[3]),
                'critical_values': {
                    '1%': float(result[4]['1%']),
                    '5%': float(result[4]['5%']),
                    '10%': float(result[4]['10%'])
                },
                'is_stationary_5pct': is_stationary_5,
                'is_stationary_1pct': is_stationary_1,
                'interpretation': interpretation,
                'max_lag_requested': max_lag,
                'max_lag_used': adjusted_max_lag,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            logger.info(f"ADF test result: p-value={result[1]:.4f}, {interpretation}")
            return adf_result
            
        except Exception as e:
            logger.error(f"❌ Error performing ADF test: {e}", exc_info=True)
            return {}
    
    def compute_rolling_correlation(self, series1: pd.Series, series2: pd.Series,
                                   window: int = 30) -> pd.Series:
        """Compute rolling correlation"""
        try:

            series1 = pd.to_numeric(series1, errors='coerce')
            series2 = pd.to_numeric(series2, errors='coerce')
            
            df = pd.DataFrame({'s1': series1, 's2': series2})
            df = df.dropna()
            
            if len(df) < window:
                logger.warning(f"Insufficient data ({len(df)}) for correlation with window={window}")
                return pd.Series(dtype=float)
            
            rolling_corr = df['s1'].rolling(window=window, min_periods=1).corr(df['s2'])
            current = rolling_corr.iloc[-1] if len(rolling_corr) > 0 and not pd.isna(rolling_corr.iloc[-1]) else 0
            logger.debug(f"Computed rolling correlation with window={window}, current={current:.3f}")
            return rolling_corr
        except Exception as e:
            logger.error(f"❌ Error computing rolling correlation: {e}", exc_info=True)
            return pd.Series(dtype=float)
    
    def compute_comprehensive_analytics(self, exchange: str, symbol1: str,
                                       symbol2: Optional[str] = None, 
                                       timeframe: str = "1m",
                                       limit: int = 1000,
                                       z_score_window: int = 20,
                                       corr_window: int = 30,
                                       regression_type: str = "OLS") -> Dict:
        """
        Compute comprehensive analytics for one or two symbols
        """
        logger.info(f"Computing comprehensive analytics: {exchange}:{symbol1}" + 
                   (f" vs {symbol2}" if symbol2 else "") + f" [{timeframe}]")
        
        result = {
            'exchange': exchange,
            'symbol1': symbol1,
            'symbol2': symbol2,
            'timeframe': timeframe,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        try:
            # Get price data for symbol1
            prices1 = self.redis.get_price_series(exchange, symbol1, timeframe, limit)
            
            if len(prices1) == 0:
                error_msg = f"No data available for {symbol1}"
                result['error'] = error_msg
                logger.warning(error_msg)
                return result
            
            logger.info(f"Retrieved {len(prices1)} price points for {symbol1}")
            
            # Convert index to ISO format strings for JSON serialization
            try:
                timestamps_list = []
                for ts in prices1.tail(500).index:
                    if isinstance(ts, pd.Timestamp):
                        timestamps_list.append(ts.isoformat())
                    elif isinstance(ts, (int, float)):
                        timestamps_list.append(pd.Timestamp(ts, unit='ms').isoformat())
                    else:
                        timestamps_list.append(str(ts))
            except Exception as e:
                logger.warning(f"Error converting timestamps: {e}")
                timestamps_list = [str(i) for i in range(len(prices1.tail(500)))]
            
            # Compute stats for symbol1
            result['symbol1_stats'] = self.compute_price_stats(prices1)
            result['prices1'] = [float(x) if not pd.isna(x) else 0 for x in prices1.tail(500).tolist()]
            result['timestamps'] = timestamps_list
            
            # If symbol2 is provided, compute pair analytics
            if symbol2:
                logger.info(f"Computing pair analytics for {symbol1} vs {symbol2}")
                prices2 = self.redis.get_price_series(exchange, symbol2, timeframe, limit)
                
                if len(prices2) > 0:
                    logger.info(f"Retrieved {len(prices2)} price points for {symbol2}")
                    result['symbol2_stats'] = self.compute_price_stats(prices2)
                    result['prices2'] = [float(x) if not pd.isna(x) else 0 for x in prices2.tail(500).tolist()]
                    
                    # OLS regression
                    ols_result = self.compute_ols_hedge_ratio(prices1, prices2, regression_type)
                    result['ols_regression'] = ols_result
                    
                    if ols_result:
                        # Compute spread
                        hedge_ratio = ols_result.get('hedge_ratio', 1.0)
                        alpha = ols_result.get('alpha', 0.0)
                        spread = self.compute_spread(prices1, prices2, hedge_ratio, alpha)
                        
                        if len(spread) > 0:
                            logger.info(f"Computed spread: {len(spread)} points")
                            result['spread_stats'] = self.compute_price_stats(spread)
                            result['spread'] = [float(x) if not pd.isna(x) else 0 for x in spread.tail(500).tolist()]
                            
                            # Z-score
                            z_score = self.compute_z_score(spread, window=z_score_window)
                            if len(z_score) > 0:
                                z_score_clean = z_score.dropna()
                                result['spread_zscore'] = [float(x) if not pd.isna(x) else 0 for x in z_score_clean.tail(500).tolist()]
                                result['current_zscore'] = float(z_score.iloc[-1]) if len(z_score) > 0 and not pd.isna(z_score.iloc[-1]) else 0
                                logger.info(f"Current Z-Score: {result['current_zscore']:.2f}")
                            
                            # ADF test
                            result['spread_adf'] = self.perform_adf_test(spread)
                        
                        # Rolling correlation
                        rolling_corr = self.compute_rolling_correlation(prices1, prices2, window=corr_window)
                        if len(rolling_corr) > 0:
                            rolling_corr_clean = rolling_corr.dropna()
                            result['rolling_correlation'] = [float(x) if not pd.isna(x) else 0 for x in rolling_corr_clean.tail(500).tolist()]
                            result['current_correlation'] = float(rolling_corr.iloc[-1]) if len(rolling_corr) > 0 and not pd.isna(rolling_corr.iloc[-1]) else 0
                            logger.info(f"Current Correlation: {result['current_correlation']:.3f}")
                else:
                    logger.warning(f"No price data available for {symbol2}")
            
            # Get volume data
            if timeframe != "tick":
                volume1 = self.redis.get_volume_series(exchange, symbol1, timeframe, limit)
                if len(volume1) > 0:
                    result['volume1'] = [float(x) if not pd.isna(x) else 0 for x in volume1.tail(500).tolist()]
                    logger.debug(f"Retrieved {len(volume1)} volume points for {symbol1}")
                    
                if symbol2:
                    volume2 = self.redis.get_volume_series(exchange, symbol2, timeframe, limit)
                    if len(volume2) > 0:
                        result['volume2'] = [float(x) if not pd.isna(x) else 0 for x in volume2.tail(500).tolist()]
                        logger.debug(f"Retrieved {len(volume2)} volume points for {symbol2}")
            
            logger.info(f"✅ Comprehensive analytics complete for {symbol1}" + (f" vs {symbol2}" if symbol2 else ""))
            return result
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"❌ Error in comprehensive analytics: {error_msg}", exc_info=True)
            import traceback
            traceback.print_exc()
            result['error'] = error_msg
            return result