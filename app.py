import asyncio
import json
import threading
import logging
import sys
from datetime import datetime, timedelta
from flask import Flask
import dash
from dash import dcc, html, Input, Output, State, ALL
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

from binance_connector import BinanceConnector
from redis_handler import RedisHandler
from analytics_engine import AnalyticsEngine
from alert_manager import AlertManager
from config_file import (
    get_redis_config,
    get_binance_config,
    get_data_config,
    get_analytics_config,
    get_dashboard_config,
    get_alert_config,
    print_config
)


def setup_logging():
    import os
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    file_handler = logging.FileHandler(
        f'logs/market_analytics_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(log_format, date_format))
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(log_format, date_format))
    
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    logging.getLogger('werkzeug').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('asyncio').setLevel(logging.WARNING)
    
    logging.info("="*70)
    logging.info("Logging system initialized")
    logging.info("="*70)


setup_logging()
logger = logging.getLogger(__name__)


class MarketAnalyticsDashboard:
    def __init__(self):
        logger.info("Initializing MarketAnalyticsDashboard")
        
        self.redis_config = get_redis_config()
        self.binance_config = get_binance_config()
        self.data_config = get_data_config()
        self.analytics_config = get_analytics_config()
        self.dashboard_config = get_dashboard_config()
        self.alert_config = get_alert_config()
        
        logger.info(f"Configuration loaded: {len(self.data_config.default_symbols)} symbols, " +
                   f"{len(self.data_config.timeframes)} timeframes")
        
        try:
            self.redis_handler = RedisHandler(
                host=self.redis_config.host,
                port=self.redis_config.port,
                db=self.redis_config.db,
                password=self.redis_config.password
            )
            logger.info("RedisHandler initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize RedisHandler: {e}")
            raise
        
        try:
            self.binance_connector = BinanceConnector(
                api_key=self.binance_config.api_key,
                api_secret=self.binance_config.api_secret
            )
            logger.info("BinanceConnector initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize BinanceConnector: {e}")
            raise
        
        try:
            self.analytics_engine = AnalyticsEngine(
                self.redis_handler,
                self.analytics_config
            )
            logger.info("AnalyticsEngine initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize AnalyticsEngine: {e}")
            raise
        
        try:
            self.alert_manager = AlertManager(
                self.analytics_engine,
                self.alert_config
            )
            logger.info("AlertManager initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize AlertManager: {e}")
            raise
        
        self.server = Flask(__name__)
        self.app = dash.Dash(
            __name__,
            server=self.server,
            external_stylesheets=[dbc.themes.CYBORG, dbc.icons.FONT_AWESOME],
            suppress_callback_exceptions=True
        )
        logger.info("Dash app created successfully")
        
        self.active_symbols = set(self.data_config.default_symbols)
        logger.info(f"Active symbols: {self.active_symbols}")
        
        self.setup_layout()
        self.setup_callbacks()
        logger.info("Dashboard layout and callbacks configured")
        
        self.alert_manager.register_callback(self.on_alert_triggered)
        self.latest_alerts = []
        logger.info("Alert callback registered")
    
    def on_alert_triggered(self, alert_event: dict):
        logger.warning(f"🔔 ALERT TRIGGERED: {alert_event['message']}")
        print(f"\n🔔 ALERT: {alert_event['message']}")
        self.latest_alerts.append(alert_event)
        self.latest_alerts = self.latest_alerts[-10:]
    
    async def handle_trade(self, trade_data: dict):

        try:
            self.redis_handler.store_trade(trade_data)
        except Exception as e:
            logger.error(f"Error handling trade: {e}")
    
    async def start_data_collection(self):

        try:
            logger.info(f"Starting data collection for symbols: {list(self.active_symbols)}")
            await self.binance_connector.subscribe_trades(
                list(self.active_symbols),
                self.handle_trade
            )
        except Exception as e:
            logger.error(f"❌ Error in data collection: {e}", exc_info=True)
            print(f"❌ Error in data collection: {e}")
    
    async def start_alert_manager(self):

        try:
            logger.info(f"Starting alert manager with check interval: {self.alert_config.check_interval}s")
            await self.alert_manager.run(check_interval=self.alert_config.check_interval)
        except Exception as e:
            logger.error(f"❌ Error in alert manager: {e}", exc_info=True)
            print(f"❌ Error in alert manager: {e}")
    
    def setup_layout(self):

        logger.info("Setting up dashboard layout")
        
        self.app.layout = dbc.Container([
            # Header
            dbc.Row([
                dbc.Col([
                    html.H1([
                        html.I(className="fas fa-chart-line me-3"),
                        "Real-Time Market Analytics"
                    ], className="text-center mb-1 mt-3"),
                    html.P("Live Trade Data · Pair Analytics · Custom Alerts", 
                          className="text-center text-muted"),
                    html.Hr()
                ])
            ]),
            
            # Control Panel
            dbc.Card([
                dbc.CardHeader([
                    html.I(className="fas fa-sliders-h me-2"),
                    "Control Panel"
                ], className="fw-bold"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Label("Symbol 1:", className="fw-bold small"),
                            dcc.Dropdown(
                                id='symbol1-dropdown',
                                options=[{'label': s, 'value': s} for s in self.data_config.default_symbols],
                                value=self.data_config.default_symbols[0],
                                clearable=False,
                                className="mb-2"
                            )
                        ], width=2),
                        
                        dbc.Col([
                            html.Label("Symbol 2:", className="fw-bold small"),
                            dcc.Dropdown(
                                id='symbol2-dropdown',
                                options=[{'label': s, 'value': s} for s in self.data_config.default_symbols] + [{'label': 'None', 'value': ''}],
                                value=self.data_config.default_symbols[1] if len(self.data_config.default_symbols) > 1 else '',
                                clearable=True,
                                className="mb-2"
                            )
                        ], width=2),
                        
                        dbc.Col([
                            html.Label("Timeframe:", className="fw-bold small"),
                            dcc.Dropdown(
                                id='timeframe-dropdown',
                                options=[{'label': tf, 'value': tf} for tf in self.data_config.timeframes],
                                value='1m',
                                clearable=False,
                                className="mb-2"
                            )
                        ], width=2),
                        
                        dbc.Col([
                            html.Label("Z-Score Window:", className="fw-bold small"),
                            dcc.Dropdown(
                                id='zscore-window-dropdown',
                                options=[{'label': str(w), 'value': w} for w in self.analytics_config.rolling_windows],
                                value=self.analytics_config.z_score_window,
                                clearable=False,
                                className="mb-2"
                            )
                        ], width=2),
                        
                        dbc.Col([
                            html.Label("Regression:", className="fw-bold small"),
                            dcc.Dropdown(
                                id='regression-dropdown',
                                options=[{'label': r, 'value': r} for r in self.analytics_config.regression_types],
                                value='OLS',
                                clearable=False,
                                className="mb-2"
                            )
                        ], width=2),
                        
                        dbc.Col([
                            html.Label("Actions:", className="fw-bold small"),
                            dbc.ButtonGroup([
                                dbc.Button("🔄", id="refresh-button", size="sm", color="primary", className="me-1"),
                                dbc.Button("ADF", id="adf-button", size="sm", color="info"),
                            ], className="w-100")
                        ], width=2),
                    ])
                ])
            ], className="mb-3"),
            
            # Status Bar
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.Div([
                                html.I(className="fas fa-circle text-success me-2"),
                                html.Span("Live", className="fw-bold")
                            ], id='status-indicator')
                        ], className="p-2 text-center")
                    ])
                ], width=2),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.Div([
                                html.I(className="fas fa-clock me-2"),
                                html.Span("--:--:--", id='last-update')
                            ], className="text-center small")
                        ], className="p-2")
                    ])
                ], width=2),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.Div([
                                html.I(className="fas fa-database me-2"),
                                html.Span("0", id='redis-keys')
                            ], className="text-center small")
                        ], className="p-2")
                    ])
                ], width=2),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.Div([
                                html.I(className="fas fa-bell me-2"),
                                html.Span("0 alerts", id='alert-count')
                            ], className="text-center small")
                        ], className="p-2")
                    ])
                ], width=3),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.Div(id='latest-alert', children="No alerts", className="text-center small text-muted")
                        ], className="p-2")
                    ])
                ], width=3),
            ], className="mb-3"),
            
            # Main Content Tabs
            dbc.Tabs([
                # Price Analysis Tab
                dbc.Tab(label="📈 Price Analysis", children=[
                    dbc.Row([
                        dbc.Col([
                            dcc.Graph(id='price-chart', style={'height': '450px'})
                        ], width=12)
                    ], className="mt-3"),
                    
                    dbc.Row([
                        dbc.Col([
                            dcc.Graph(id='volume-chart', style={'height': '250px'})
                        ], width=6),
                        
                        dbc.Col([
                            html.Div(id='price-stats-cards')
                        ], width=6)
                    ], className="mt-2")
                ]),
                
                # Pair Analytics Tab
                dbc.Tab(label="🔗 Pair Analytics", children=[
                    dbc.Row([
                        dbc.Col([
                            html.Div(id='pair-summary-cards')
                        ], width=12)
                    ], className="mt-3"),
                    
                    dbc.Row([
                        dbc.Col([
                            dcc.Graph(id='spread-chart', style={'height': '350px'})
                        ], width=12)
                    ], className="mt-2"),
                    
                    dbc.Row([
                        dbc.Col([
                            dcc.Graph(id='zscore-chart', style={'height': '300px'})
                        ], width=6),
                        
                        dbc.Col([
                            dcc.Graph(id='correlation-chart', style={'height': '300px'})
                        ], width=6)
                    ], className="mt-2")
                ]),
                
                # Statistical Tests Tab
                dbc.Tab(label="📊 Statistical Tests", children=[
                    dbc.Row([
                        dbc.Col([
                            html.Div(id='ols-card')
                        ], width=6),
                        
                        dbc.Col([
                            html.Div(id='adf-card')
                        ], width=6)
                    ], className="mt-3"),
                    
                    dbc.Row([
                        dbc.Col([
                            dcc.Graph(id='scatter-plot', style={'height': '400px'})
                        ], width=6),
                        
                        dbc.Col([
                            dcc.Graph(id='residuals-plot', style={'height': '400px'})
                        ], width=6)
                    ], className="mt-2")
                ]),
                
                # Alerts Tab
                dbc.Tab(label="🔔 Alerts", children=[
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("Create New Alert", className="fw-bold"),
                                dbc.CardBody([
                                    dbc.Row([
                                        dbc.Col([
                                            html.Label("Alert Type:", className="small"),
                                            dcc.Dropdown(
                                                id='alert-type-dropdown',
                                                options=[
                                                    {'label': 'Z-Score', 'value': 'zscore'},
                                                    {'label': 'Price', 'value': 'price'},
                                                    {'label': 'Correlation', 'value': 'correlation'},
                                                    {'label': 'Price Change %', 'value': 'price_change'}
                                                ],
                                                value='zscore',
                                                className="mb-2"
                                            )
                                        ], width=3),
                                        
                                        dbc.Col([
                                            html.Label("Condition:", className="small"),
                                            dcc.Dropdown(
                                                id='alert-condition-dropdown',
                                                options=[
                                                    {'label': '>', 'value': '>'},
                                                    {'label': '<', 'value': '<'},
                                                    {'label': '>=', 'value': '>='},
                                                    {'label': '<=', 'value': '<='}
                                                ],
                                                value='>',
                                                className="mb-2"
                                            )
                                        ], width=2),
                                        
                                        dbc.Col([
                                            html.Label("Threshold:", className="small"),
                                            dbc.Input(id='alert-threshold-input', type='number', 
                                                     value=2.0, step=0.1, className="mb-2")
                                        ], width=2),
                                        
                                        dbc.Col([
                                            html.Label("Timeframe:", className="small"),
                                            dcc.Dropdown(
                                                id='alert-timeframe-dropdown',
                                                options=[{'label': tf, 'value': tf} for tf in self.data_config.timeframes],
                                                value='1m',
                                                className="mb-2"
                                            )
                                        ], width=2),
                                        
                                        dbc.Col([
                                            html.Label(" ", className="small"),
                                            dbc.Button("Add Alert", id='add-alert-button', 
                                                      color="success", className="w-100")
                                        ], width=3)
                                    ])
                                ])
                            ])
                        ], width=12)
                    ], className="mt-3"),
                    
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("Active Alerts", className="fw-bold"),
                                dbc.CardBody([
                                    html.Div(id='alerts-list')
                                ])
                            ])
                        ], width=6),
                        
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("Alert History", className="fw-bold"),
                                dbc.CardBody([
                                    html.Div(id='alert-history')
                                ])
                            ])
                        ], width=6)
                    ], className="mt-3")
                ])
            ]),
            
            # Hidden data stores
            dcc.Store(id='analytics-store'),
            dcc.Store(id='alerts-store', data=[]),
            
            # Interval components
            dcc.Interval(id='fast-interval', interval=2000, n_intervals=0),
            dcc.Interval(id='medium-interval', interval=2000, n_intervals=0),
            dcc.Interval(id='slow-interval', interval=5000, n_intervals=0),
            
        ], fluid=True, className="p-3", style={'backgroundColor': '#0a0a0a'})
        
        logger.info("Dashboard layout setup complete")
    
    def setup_callbacks(self):
        logger.info("Setting up dashboard callbacks")

        @self.app.callback(
            Output('analytics-store', 'data'),
            [Input('fast-interval', 'n_intervals'),
             Input('refresh-button', 'n_clicks')],
            [State('symbol1-dropdown', 'value'),
             State('symbol2-dropdown', 'value'),
             State('timeframe-dropdown', 'value'),
             State('zscore-window-dropdown', 'value'),
             State('regression-dropdown', 'value')]
        )
        def fetch_analytics(n_fast, n_refresh, symbol1, symbol2, timeframe, z_window, regression):
            try:
                analytics = self.analytics_engine.compute_comprehensive_analytics(
                    exchange='BINANCE',
                    symbol1=symbol1,
                    symbol2=symbol2 if symbol2 else None,
                    timeframe=timeframe,
                    limit=100000,
                    z_score_window=z_window,
                    corr_window=self.analytics_config.correlation_window,
                    regression_type=regression
                )
                return json.dumps(analytics, default=str)
            except Exception as e:
                logger.error(f"Error fetching analytics: {e}", exc_info=True)
                return json.dumps({'error': str(e)})
        
        @self.app.callback(
            [Output('status-indicator', 'children'),
             Output('last-update', 'children'),
             Output('redis-keys', 'children'),
             Output('alert-count', 'children'),
             Output('latest-alert', 'children')],
            [Input('fast-interval', 'n_intervals')]
        )
        def update_status(n):
            try:
                stats = self.redis_handler.get_stats()
                now = datetime.now().strftime("%H:%M:%S")
                keys = stats.get('total_keys', 0)
                
                history = self.alert_manager.get_alert_history()
                alert_text = "No alerts"
                if history:
                    last = history[-1]
                    alert_text = f"⚠️ {last['type']}: {last['actual_value']:.2f}"
                
                return (
                    [html.I(className="fas fa-circle text-success me-2"), html.Span("Live", className="fw-bold")],
                    now,
                    str(keys),
                    f"{len(self.alert_manager.get_alerts())} active",
                    alert_text
                )
            except Exception as e:
                logger.error(f"Error updating status: {e}")
                return (
                    [html.I(className="fas fa-circle text-danger me-2"), html.Span("Error")],
                    "--:--:--", "0", "0", "Error"
                )
        
        # Price chart with proper scrolling
        @self.app.callback(
            Output('price-chart', 'figure'),
            [Input('analytics-store', 'data')]
        )
        def update_price_chart(analytics_json):
            try:
                analytics = json.loads(analytics_json)
                
                if 'error' in analytics:
                    fig = go.Figure()
                    fig.add_annotation(text=f"Error: {analytics['error']}", 
                                      xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
                    return fig
                
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                
                # Get data
                prices1 = analytics.get('prices1', [])
                prices2 = analytics.get('prices2', [])
                timestamps = analytics.get('timestamps', [])
                
                if not prices1 or not timestamps:
                    return go.Figure()
                
                # Ensure timestamps and prices have same length
                min_len = min(len(timestamps), len(prices1))
                timestamps = timestamps[-min_len:]
                prices1 = prices1[-min_len:]
                
                # Add symbol1 trace
                fig.add_trace(
                    go.Scatter(
                        x=timestamps, 
                        y=prices1, 
                        name=analytics.get('symbol1', 'Symbol1'),
                        line=dict(color='#00ff41', width=2),
                        mode='lines'
                    ),
                    secondary_y=False
                )
                
                # Add symbol2 if exists
                if prices2:
                    min_len2 = min(len(timestamps), len(prices2))
                    timestamps2 = timestamps[-min_len2:]
                    prices2 = prices2[-min_len2:]
                    
                    fig.add_trace(
                        go.Scatter(
                            x=timestamps2, 
                            y=prices2, 
                            name=analytics.get('symbol2', 'Symbol2'),
                            line=dict(color='#ff006e', width=2),
                            mode='lines'
                        ),
                        secondary_y=True
                    )
                
                # Configure layout with auto-scrolling
                fig.update_layout(
                    title=f"Price Chart - {analytics.get('symbol1', '')}" + 
                          (f" vs {analytics.get('symbol2', '')}" if prices2 else ""),
                    template='plotly_dark',
                    height=450,
                    hovermode='x unified',
                    showlegend=True,
                    paper_bgcolor='#0a0a0a',
                    plot_bgcolor='#1a1a1a',
                    xaxis=dict(
                        rangeslider=dict(visible=False),
                        type='date'
                    )
                )
                
                fig.update_yaxes(title_text=analytics.get('symbol1', ''), secondary_y=False)
                if prices2:
                    fig.update_yaxes(title_text=analytics.get('symbol2', ''), secondary_y=True)
                
                return fig
                
            except Exception as e:
                logger.error(f"Error updating price chart: {e}", exc_info=True)
                fig = go.Figure()
                fig.add_annotation(text=f"Chart Error: {str(e)}", 
                                  xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
                return fig
        
        # Volume chart
        @self.app.callback(
            Output('volume-chart', 'figure'),
            [Input('analytics-store', 'data')]
        )
        def update_volume_chart(analytics_json):
            try:
                analytics = json.loads(analytics_json)
                
                fig = go.Figure()
                
                volume = analytics.get('volume1', [])
                timestamps = analytics.get('timestamps', [])
                
                if volume and timestamps:
                    min_len = min(len(timestamps), len(volume))
                    timestamps = timestamps[-min_len:]
                    volume = volume[-min_len:]
                    
                    fig.add_trace(go.Bar(
                        x=timestamps, 
                        y=volume,
                        name='Volume',
                        marker_color='#00d9ff'
                    ))
                
                fig.update_layout(
                    title="Volume",
                    template='plotly_dark',
                    height=250,
                    paper_bgcolor='#0a0a0a',
                    plot_bgcolor='#1a1a1a',
                    xaxis=dict(type='date')
                )
                
                return fig
            except Exception as e:
                logger.error(f"Error updating volume chart: {e}")
                return go.Figure()
        
        # Price stats cards
        @self.app.callback(
            Output('price-stats-cards', 'children'),
            [Input('analytics-store', 'data')]
        )
        def update_price_stats(analytics_json):
            try:
                analytics = json.loads(analytics_json)
                stats = analytics.get('symbol1_stats', {})
                
                if not stats:
                    return html.Div("No data available", className="text-muted")
                
                cards = dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H6("Current Price", className="text-muted small"),
                                html.H4(f"${stats.get('current_price', 0):,.2f}", className="text-success mb-0")
                            ])
                        ])
                    ], width=6, className="mb-2"),
                    
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H6("Change %", className="text-muted small"),
                                html.H4(f"{stats.get('change_pct', 0):+.2f}%", 
                                       className="text-warning mb-0")
                            ])
                        ])
                    ], width=6, className="mb-2"),
                    
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H6("Volatility", className="text-muted small"),
                                html.H4(f"{stats.get('std', 0):,.2f}", className="text-info mb-0")
                            ])
                        ])
                    ], width=6, className="mb-2"),
                    
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H6("Sharpe Ratio", className="text-muted small"),
                                html.H4(f"{stats.get('sharpe_ratio', 0):.2f}", className="text-primary mb-0")
                            ])
                        ])
                    ], width=6, className="mb-2"),
                ])
                
                return cards
            except Exception as e:
                logger.error(f"Error updating price stats: {e}")
                return html.Div("Error loading stats", className="text-danger")
        
        # Pair summary cards
        @self.app.callback(
            Output('pair-summary-cards', 'children'),
            [Input('analytics-store', 'data')]
        )
        def update_pair_summary(analytics_json):
            try:
                analytics = json.loads(analytics_json)
                
                if not analytics.get('symbol2'):
                    return dbc.Alert("Select Symbol 2 for pair analytics", color="info")
                
                ols = analytics.get('ols_regression', {})
                
                cards = dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H6("Hedge Ratio (β)", className="small text-muted"),
                                html.H3(f"{ols.get('hedge_ratio', 0):.4f}", className="text-primary mb-1"),
                                html.Small(f"R² = {ols.get('r_squared', 0):.4f}", className="text-muted")
                            ])
                        ])
                    ], width=3),
                    
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H6("Current Z-Score", className="small text-muted"),
                                html.H3(f"{analytics.get('current_zscore', 0):.2f}", 
                                       className="text-warning mb-0")
                            ])
                        ])
                    ], width=3),
                    
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H6("Correlation", className="small text-muted"),
                                html.H3(f"{analytics.get('current_correlation', 0):.3f}", 
                                       className="text-info mb-0")
                            ])
                        ])
                    ], width=3),
                    
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H6("Spread σ", className="small text-muted"),
                                html.H3(f"{analytics.get('spread_stats', {}).get('std', 0):.2f}", 
                                       className="text-danger mb-0")
                            ])
                        ])
                    ], width=3),
                ])
                
                return cards
            except Exception as e:
                logger.error(f"Error updating pair summary: {e}")
                return html.Div("Error loading data")
        
        # Spread chart with proper scrolling
        @self.app.callback(
            Output('spread-chart', 'figure'),
            [Input('analytics-store', 'data')]
        )
        def update_spread_chart(analytics_json):
            try:
                analytics = json.loads(analytics_json)
                
                if not analytics.get('symbol2'):
                    return go.Figure()
                
                fig = go.Figure()
                
                ols = analytics.get('ols_regression', {})
                residuals = ols.get('residuals', [])
                timestamps = ols.get('timestamps', [])
                
                if residuals and timestamps:
                    min_len = min(len(timestamps), len(residuals))
                    timestamps = timestamps[-min_len:]
                    residuals = residuals[-min_len:]
                    
                    mean = np.mean(residuals)
                    std = np.std(residuals)
                    
                    fig.add_trace(go.Scatter(
                        x=timestamps, 
                        y=residuals,
                        name='Spread', 
                        line=dict(color='#00ff41', width=2),
                        mode='lines'
                    ))
                    
                    # Add reference lines
                    fig.add_hline(y=mean, line_dash="dash", line_color="yellow", 
                                 annotation_text="Mean", annotation_position="right")
                    fig.add_hline(y=mean+std, line_dash="dot", line_color="red", 
                                 annotation_text="+1σ", annotation_position="right")
                    fig.add_hline(y=mean-std, line_dash="dot", line_color="red", 
                                 annotation_text="-1σ", annotation_position="right")
                    fig.add_hline(y=mean+2*std, line_dash="dot", line_color="darkred", 
                                 annotation_text="+2σ", annotation_position="right")
                    fig.add_hline(y=mean-2*std, line_dash="dot", line_color="darkred", 
                                 annotation_text="-2σ", annotation_position="right")
                
                fig.update_layout(
                    title="Spread (OLS Residuals)",
                    template='plotly_dark',
                    height=350,
                    hovermode='x unified',
                    paper_bgcolor='#0a0a0a',
                    plot_bgcolor='#1a1a1a',
                    xaxis=dict(type='date')
                )
                
                return fig
            except Exception as e:
                logger.error(f"Error updating spread chart: {e}")
                return go.Figure()
        
        # Z-score chart with proper data alignment
        @self.app.callback(
            Output('zscore-chart', 'figure'),
            [Input('analytics-store', 'data')]
        )
        def update_zscore_chart(analytics_json):
            try:
                analytics = json.loads(analytics_json)
                
                if not analytics.get('symbol2'):
                    return go.Figure()
                
                fig = go.Figure()
                
                z_scores = analytics.get('spread_zscore', [])
                timestamps = analytics.get('timestamps', [])
                
                if z_scores and timestamps:
                    # Align timestamps with z_scores
                    min_len = min(len(timestamps), len(z_scores))
                    timestamps = timestamps[-min_len:]
                    z_scores = z_scores[-min_len:]
                    
                    fig.add_trace(go.Scatter(
                        x=timestamps, 
                        y=z_scores,
                        name='Z-Score',
                        line=dict(color='#ff006e', width=2),
                        fill='tozeroy',
                        mode='lines'
                    ))
                    
                    # Threshold lines
                    fig.add_hline(y=2, line_dash="dash", line_color="red", 
                                 annotation_text="Entry", annotation_position="right")
                    fig.add_hline(y=-2, line_dash="dash", line_color="red",
                                 annotation_text="Entry", annotation_position="right")
                    fig.add_hline(y=0, line_color="white", line_width=1)
                
                fig.update_layout(
                    title=f"Z-Score (Live) - Current: {analytics.get('current_zscore', 0):.2f}",
                    template='plotly_dark',
                    height=300,
                    hovermode='x unified',
                    paper_bgcolor='#0a0a0a',
                    plot_bgcolor='#1a1a1a',
                    xaxis=dict(type='date')
                )
                
                return fig
            except Exception as e:
                logger.error(f"Error updating z-score chart: {e}")
                return go.Figure()
        
        # Correlation chart
        @self.app.callback(
            Output('correlation-chart', 'figure'),
            [Input('analytics-store', 'data')]
        )
        def update_correlation_chart(analytics_json):
            try:
                analytics = json.loads(analytics_json)
                
                if not analytics.get('symbol2'):
                    return go.Figure()
                
                fig = go.Figure()
                
                corr = analytics.get('rolling_correlation', [])
                timestamps = analytics.get('timestamps', [])
                
                if corr and timestamps:
                    min_len = min(len(timestamps), len(corr))
                    timestamps = timestamps[-min_len:]
                    corr = corr[-min_len:]
                    
                    fig.add_trace(go.Scatter(
                        x=timestamps, 
                        y=corr,
                        name='Correlation',
                        line=dict(color='#00d9ff', width=2),
                        fill='tozeroy',
                        mode='lines'
                    ))
                
                fig.update_layout(
                    title=f"Rolling Correlation - Current: {analytics.get('current_correlation', 0):.3f}",
                    template='plotly_dark',
                    height=300,
                    yaxis=dict(range=[-1, 1]),
                    hovermode='x unified',
                    paper_bgcolor='#0a0a0a',
                    plot_bgcolor='#1a1a1a',
                    xaxis=dict(type='date')
                )
                
                return fig
            except Exception as e:
                logger.error(f"Error updating correlation chart: {e}")
                return go.Figure()
        
        # OLS card
        @self.app.callback(
            Output('ols-card', 'children'),
            [Input('analytics-store', 'data')]
        )
        def update_ols_card(analytics_json):
            try:
                analytics = json.loads(analytics_json)
                
                if not analytics.get('symbol2'):
                    return dbc.Alert("Select Symbol 2 for regression", color="info")
                
                ols = analytics.get('ols_regression', {})
                
                if not ols:
                    return dbc.Alert("No regression data available", color="warning")
                
                card = dbc.Card([
                    dbc.CardHeader("OLS Regression Results", className="fw-bold"),
                    dbc.CardBody([
                        html.Table([
                            html.Tr([html.Td("Type:"), html.Td(ols.get('regression_type', 'OLS'), className="fw-bold")]),
                            html.Tr([html.Td("Hedge Ratio (β):"), html.Td(f"{ols.get('hedge_ratio', 0):.6f}")]),
                            html.Tr([html.Td("Alpha (α):"), html.Td(f"{ols.get('alpha', 0):.6f}")]),
                            html.Tr([html.Td("R-squared:"), html.Td(f"{ols.get('r_squared', 0):.4f}")]),
                            html.Tr([html.Td("Spread Mean:"), html.Td(f"{ols.get('spread_mean', 0):.4f}")]),
                            html.Tr([html.Td("Spread Std:"), html.Td(f"{ols.get('spread_std', 0):.4f}")]),
                        ], className="table table-sm table-dark")
                    ])
                ])
                
                return card
            except Exception as e:
                logger.error(f"Error updating OLS card: {e}")
                return dbc.Alert("Error", color="danger")
        
        # ADF card
        @self.app.callback(
            Output('adf-card', 'children'),
            [Input('adf-button', 'n_clicks')],
            [State('analytics-store', 'data')]
        )
        def update_adf_card(n_clicks, analytics_json):
            try:
                if not n_clicks:
                    return dbc.Alert("Click 'ADF' button to run stationarity test", color="info")
                
                analytics = json.loads(analytics_json)
                adf = analytics.get('spread_adf', {})
                
                if not adf:
                    return dbc.Alert("No ADF test results available", color="warning")
                
                card = dbc.Card([
                    dbc.CardHeader("ADF Stationarity Test", className="fw-bold"),
                    dbc.CardBody([
                        html.Table([
                            html.Tr([html.Td("ADF Statistic:"), html.Td(f"{adf.get('adf_statistic', 0):.4f}")]),
                            html.Tr([html.Td("P-value:"), html.Td(f"{adf.get('p_value', 0):.4f}")]),
                            html.Tr([html.Td("Lags Used:"), html.Td(f"{adf.get('used_lag', 0)}")]),
                            html.Tr([html.Td("Observations:"), html.Td(f"{adf.get('n_observations', 0)}")]),
                        ], className="table table-sm table-dark mb-2"),
                        html.H6("Critical Values:", className="mb-2"),
                        html.Table([
                            html.Tr([html.Td("1%:"), html.Td(f"{adf.get('critical_values', {}).get('1%', 0):.4f}")]),
                            html.Tr([html.Td("5%:"), html.Td(f"{adf.get('critical_values', {}).get('5%', 0):.4f}")]),
                            html.Tr([html.Td("10%:"), html.Td(f"{adf.get('critical_values', {}).get('10%', 0):.4f}")]),
                        ], className="table table-sm table-dark mb-2"),
                        html.Hr(),
                        dbc.Alert(
                            adf.get('interpretation', 'Unknown'),
                            color="success" if adf.get('is_stationary_5pct') else "warning"
                        )
                    ])
                ])
                
                return card
            except Exception as e:
                logger.error(f"Error updating ADF card: {e}")
                return dbc.Alert("Error running ADF test", color="danger")
        
        # Scatter plot
        @self.app.callback(
            Output('scatter-plot', 'figure'),
            [Input('analytics-store', 'data')]
        )
        def update_scatter(analytics_json):
            try:
                analytics = json.loads(analytics_json)
                
                if not analytics.get('symbol2'):
                    return go.Figure()
                
                prices1 = analytics.get('prices1', [])
                prices2 = analytics.get('prices2', [])
                ols = analytics.get('ols_regression', {})
                
                fig = go.Figure()
                
                if prices1 and prices2:
                    min_len = min(len(prices1), len(prices2))
                    p1 = prices1[-min_len:]
                    p2 = prices2[-min_len:]
                    
                    fig.add_trace(go.Scatter(
                        x=p2, y=p1,
                        mode='markers',
                        name='Data Points',
                        marker=dict(size=4, color='cyan', opacity=0.5)
                    ))
                    
                    # Regression line
                    if ols:
                        hedge = ols.get('hedge_ratio', 1)
                        alpha = ols.get('alpha', 0)
                        x_line = np.linspace(min(p2), max(p2), 100)
                        y_line = alpha + hedge * x_line
                        
                        fig.add_trace(go.Scatter(
                            x=x_line, y=y_line,
                            mode='lines',
                            name=f'y = {alpha:.2f} + {hedge:.4f}x',
                            line=dict(color='red', width=2)
                        ))
                
                fig.update_layout(
                    title="Regression Scatter Plot",
                    xaxis_title=analytics.get('symbol2', 'Symbol 2'),
                    yaxis_title=analytics.get('symbol1', 'Symbol 1'),
                    template='plotly_dark',
                    height=400,
                    paper_bgcolor='#0a0a0a',
                    plot_bgcolor='#1a1a1a'
                )
                
                return fig
            except Exception as e:
                logger.error(f"Error updating scatter plot: {e}")
                return go.Figure()
        
        # Residuals plot
        @self.app.callback(
            Output('residuals-plot', 'figure'),
            [Input('analytics-store', 'data')]
        )
        def update_residuals(analytics_json):
            try:
                analytics = json.loads(analytics_json)
                
                if not analytics.get('symbol2'):
                    return go.Figure()
                
                ols = analytics.get('ols_regression', {})
                fitted = ols.get('fitted_values', [])
                residuals = ols.get('residuals', [])
                
                fig = go.Figure()
                
                if fitted and residuals:
                    min_len = min(len(fitted), len(residuals))
                    fitted = fitted[-min_len:]
                    residuals = residuals[-min_len:]
                    
                    fig.add_trace(go.Scatter(
                        x=fitted, y=residuals,
                        mode='markers',
                        name='Residuals',
                        marker=dict(size=4, color='lime', opacity=0.5)
                    ))
                    
                    fig.add_hline(y=0, line_color="red", line_dash="dash")
                    
                    # Add std bands
                    if residuals:
                        std_res = np.std(residuals)
                        fig.add_hline(y=std_res, line_color="orange", line_dash="dot", 
                                     annotation_text="+1σ")
                        fig.add_hline(y=-std_res, line_color="orange", line_dash="dot",
                                     annotation_text="-1σ")
                
                fig.update_layout(
                    title="Residuals vs Fitted Values",
                    xaxis_title="Fitted Values",
                    yaxis_title="Residuals",
                    template='plotly_dark',
                    height=400,
                    paper_bgcolor='#0a0a0a',
                    plot_bgcolor='#1a1a1a'
                )
                
                return fig
            except Exception as e:
                logger.error(f"Error updating residuals plot: {e}")
                return go.Figure()
        
        # Add alert
        @self.app.callback(
            Output('alerts-store', 'data'),
            [Input('add-alert-button', 'n_clicks')],
            [State('symbol1-dropdown', 'value'),
             State('symbol2-dropdown', 'value'),
             State('alert-type-dropdown', 'value'),
             State('alert-condition-dropdown', 'value'),
             State('alert-threshold-input', 'value'),
             State('alert-timeframe-dropdown', 'value'),
             State('alerts-store', 'data')]
        )
        def add_alert(n_clicks, sym1, sym2, alert_type, condition, threshold, timeframe, current_alerts):
            if not n_clicks:
                return current_alerts or []
            
            try:
                alert_config = {
                    'id': f"alert_{datetime.now().timestamp()}",
                    'exchange': 'BINANCE',
                    'symbol1': sym1,
                    'symbol2': sym2 if sym2 else None,
                    'type': alert_type,
                    'condition': condition,
                    'threshold': float(threshold),
                    'timeframe': timeframe,
                    'enabled': True
                }
                
                self.alert_manager.add_alert(alert_config)
                logger.info(f"Alert added: {alert_config}")
                
                if not current_alerts:
                    current_alerts = []
                current_alerts.append(alert_config)
                
                return current_alerts
            except Exception as e:
                logger.error(f"Error adding alert: {e}")
                return current_alerts or []
        
        # Alerts list
        @self.app.callback(
            Output('alerts-list', 'children'),
            [Input('alerts-store', 'data'),
             Input('medium-interval', 'n_intervals')]
        )
        def update_alerts_list(alerts, n):
            if not alerts:
                return html.P("No active alerts", className="text-muted")
            
            items = []
            for alert in alerts:
                items.append(
                    dbc.ListGroupItem([
                        html.Div([
                            html.Strong(f"{alert['type'].upper()}: "),
                            f"{alert['symbol1']} {alert['condition']} {alert['threshold']} ",
                            html.Small(f"({alert['timeframe']})", className="text-muted")
                        ])
                    ])
                )
            
            return dbc.ListGroup(items)
        
        # Alert history
        @self.app.callback(
            Output('alert-history', 'children'),
            [Input('fast-interval', 'n_intervals')]
        )
        def update_alert_history(n):
            history = self.alert_manager.get_alert_history()
            
            if not history:
                return html.P("No alerts triggered yet", className="text-muted")
            
            items = []
            for alert in reversed(history[-10:]):
                items.append(
                    dbc.ListGroupItem([
                        html.Div([
                            html.Strong(alert['message']),
                            html.Br(),
                            html.Small(alert['triggered_at'], className="text-muted")
                        ])
                    ], color="warning")
                )
            
            return dbc.ListGroup(items)
        
        logger.info("Dashboard callbacks setup complete")
    
    def run_async_tasks(self):
        logger.info("Starting async tasks thread")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            loop.run_until_complete(asyncio.gather(
                self.start_data_collection(),
                self.start_alert_manager()
            ))
        except Exception as e:
            logger.error(f"❌ Error in async tasks: {e}", exc_info=True)
            print(f"❌ Error in async tasks: {e}")
        finally:
            loop.close()
            logger.info("Async tasks loop closed")
    
    def run(self):
        print_config()
        
        logger.info("="*70)
        logger.info("🚀 Starting Real-Time Market Analytics Dashboard")
        logger.info("="*70)
        
        print(f"\n{'='*70}")
        print("🚀 Starting Real-Time Market Analytics Dashboard")
        print(f"{'='*70}\n")
        print(f"📊 Dashboard URL: http://{self.dashboard_config.host}:{self.dashboard_config.port}")
        print(f"🔔 Alert checking interval: {self.alert_config.check_interval}s")
        print(f"⚡ Fast updates: {self.dashboard_config.fast_update_interval}ms")
        print(f"📝 Logs directory: ./logs/")
        print(f"\n{'='*70}\n")
        
        async_thread = threading.Thread(target=self.run_async_tasks, daemon=True)
        async_thread.start()
        logger.info("Async tasks thread started")
        
        logger.info(f"Starting Dash server on {self.dashboard_config.host}:{self.dashboard_config.port}")
        self.app.run(
            host=self.dashboard_config.host,
            port=self.dashboard_config.port,
            debug=self.dashboard_config.debug
        )


def main():
    try:
        logger.info("="*70)
        logger.info("APPLICATION STARTUP")
        logger.info("="*70)
        
        dashboard = MarketAnalyticsDashboard()
        dashboard.run()
        
    except KeyboardInterrupt:
        logger.info("Application interrupted by user (Ctrl+C)")
        print("\n\n👋 Shutting down gracefully...")
    except Exception as e:
        logger.error(f"Fatal error in main: {e}", exc_info=True)
        print(f"\n❌ Fatal error: {e}")
        raise
    finally:
        logger.info("="*70)
        logger.info("APPLICATION SHUTDOWN")
        logger.info("="*70)


if __name__ == "__main__":
    main()