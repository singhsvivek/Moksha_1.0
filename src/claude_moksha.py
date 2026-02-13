"""
MOKSHA CAPITAL MANAGEMENT - OPTIMIZED QUARTERLY SILVER BACKTEST
Production-Ready Implementation v2.0

Strategy: AGQ Intraday Mean-Reversion Trap Pattern Recognition
Author: Quantitative Research Team
Date: February 2026

This implementation includes all four phases of optimization:
    Phase 1: Infrastructure (Transaction costs, ATR stops, position sizing)
    Phase 2: Signal Quality (Volume confirmation, time filters, gap detection)
    Phase 3: Risk Management (Kelly Criterion, dynamic scaling, loss limits)
    Phase 4: Statistical Analysis (Monte Carlo, regime detection, MAE/MFE)
"""

import pandas as pd
import numpy as np
import alpaca_trade_api as tradeapi
from datetime import datetime, timedelta
from tqdm import tqdm
import sys
import json
from scipy import stats
from sklearn.cluster import KMeans
from Moksha_1.config import settings
from Moksha_1.utils.logger import logger

# ============================================================================
# CONFIGURATION PARAMETERS
# ============================================================================

SYMBOL = 'AGQ'                  # ProShares Ultra Silver ETF (2x leverage)
START_DATE = '2024-01-01'       # Extended backtest period
CAPITAL = 25000.0               # Starting capital
BASE_RISK_PCT = 0.02            # Base risk per trade (2% - conservative for leverage)
MAX_RISK_PCT = 0.03             # Maximum risk per trade
MIN_POSITION_SIZE = 10          # Minimum shares to trade
ATR_PERIOD = 14                 # ATR calculation period
ATR_STOP_MULTIPLIER = 1.5       # Stop loss = entry ± (ATR × multiplier)
TAKE_PROFIT_RATIO = 2.0         # Take profit at 2R
BREAKEVEN_RATIO = 1.0           # Move stop to breakeven at 1R

# Transaction cost model (basis points + fixed spread)
SLIPPAGE_BPS = 1.5              # 1.5 basis points per side
SPREAD_DOLLARS = 0.02           # $0.02 spread per share
COMMISSION = 0.0                # Zero commission (most brokers now)

# Signal quality filters
MIN_VOLUME_MULTIPLIER = 1.2     # Q2 entry volume must be 1.2x Q1 average
MIN_Q1_BARS = 15                # Minimum bars in Q1 for valid range
MAX_DAILY_TRADES = 3            # Maximum trades per day (avoid overtrading)

# Risk management
MAX_CONSECUTIVE_LOSSES = 2      # Reduce size after this many losses
POSITION_SCALE_DOWN = 0.50      # Scale to 50% after consecutive losses
MAX_DAILY_LOSS_PCT = 0.04       # Stop trading day after 4% account loss
KELLY_FRACTION = 0.25           # Use quarter-Kelly for safety

# Statistical analysis
MONTE_CARLO_RUNS = 5000         # Monte Carlo simulation iterations
REGIME_LOOKBACK = 20            # Days for regime detection
MIN_TRADES_FOR_KELLY = 20       # Minimum trades before applying Kelly

# ============================================================================
# CORE BACKTEST ENGINE
# ============================================================================

class MokshaOptimizedBacktest:
    """
    Production-grade backtest engine for AGQ intraday mean-reversion strategy.
    Implements institutional-quality risk management and statistical analysis.
    """
    
    def __init__(self):
        """Initialize Alpaca connection and data structures."""
        try:
            base_url = getattr(settings, 'ALPACA_BASE_URL', "https://paper-api.alpaca.markets")
            self.api = tradeapi.REST(
                settings.ALPACA_API_KEY, 
                settings.ALPACA_SECRET_KEY, 
                base_url=base_url, 
                api_version='v2'
            )
            logger.info("✅ Alpaca API Connected Successfully")
        except Exception as e:
            logger.error(f"❌ Alpaca Connection Failed: {e}")
            sys.exit(1)
        
        # Trade tracking for dynamic risk management
        self.trade_history = []
        self.consecutive_losses = 0
        self.daily_pnl = {}
        self.current_position_scale = 1.0
        
        # Statistical analysis storage
        self.mae_mfe_data = []  # Maximum Adverse/Favorable Excursion
        self.regime_data = []
        
    def calculate_transaction_costs(self, shares, entry_price):
        """
        Calculate realistic transaction costs including slippage and spread.
        
        Args:
            shares: Number of shares
            entry_price: Entry price per share
            
        Returns:
            Total round-trip transaction cost in dollars
        """
        position_value = shares * entry_price
        slippage_cost = (position_value * SLIPPAGE_BPS / 10000) * 2  # Round trip
        spread_cost = shares * SPREAD_DOLLARS * 2  # Round trip
        commission_cost = COMMISSION * 2  # Round trip
        
        total_cost = slippage_cost + spread_cost + commission_cost
        return total_cost
    
    def calculate_atr(self, df, period=ATR_PERIOD):
        """
        Calculate Average True Range for volatility-adjusted stops.
        
        The ATR gives us a dynamic measure of volatility. In calm markets,
        our stops will be tighter. In volatile markets, they'll be wider.
        This prevents getting stopped out by normal market noise.
        """
        high = df['high']
        low = df['low']
        close = df['close']
        
        # True Range calculation
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr
    
    def fetch_real_data(self):
        """
        Fetch historical bar data from Alpaca.
        Extended to full year for better statistical significance.
        """
        logger.info(f"⏳ Fetching Historical Data for {SYMBOL}...")
        end_dt = datetime.now()
        start_dt = end_dt - timedelta(days=730)  # 2 years for regime detection
        
        try:
            bars = self.api.get_bars(
                SYMBOL, 
                '5Min', 
                start=start_dt.strftime('%Y-%m-%d'), 
                end=end_dt.strftime('%Y-%m-%d'), 
                feed='iex'
            ).df
            
            if bars.empty:
                raise ValueError("No data returned from Alpaca")
            
            # Timezone handling
            if bars.index.tz is None:
                bars.index = bars.index.tz_localize('UTC')
            bars.index = bars.index.tz_convert('America/Chicago')
            
            # Calculate ATR for all bars
            bars['atr'] = self.calculate_atr(bars)
            
            logger.info(f"✅ Fetched {len(bars)} bars from {bars.index[0]} to {bars.index[-1]}")
            return bars
            
        except Exception as e:
            logger.exception(f"❌ Data Fetch Error: {e}")
            return pd.DataFrame()
    
    def detect_market_regime(self, df, current_date):
        """
        Detect market regime using volatility and trend clustering.
        
        We classify markets into three regimes:
        1. Low Volatility Mean-Reverting (best for our strategy)
        2. High Volatility Choppy (moderate performance)
        3. Trending (worst for mean-reversion)
        
        This helps us understand when to be more aggressive or defensive.
        """
        lookback_start = current_date - timedelta(days=REGIME_LOOKBACK)
        recent_data = df[df.index.date >= lookback_start.date()]
        
        if len(recent_data) < 50:
            return "unknown", 1.0
        
        # Calculate regime features
        returns = recent_data['close'].pct_change()
        volatility = returns.std()
        trend_strength = abs(returns.mean() / returns.std()) if returns.std() > 0 else 0
        
        # Simple regime classification
        if volatility < returns.std() * 0.8 and trend_strength < 0.3:
            return "mean_reverting", 1.2  # Favorable regime, increase size
        elif volatility > returns.std() * 1.5:
            return "high_volatility", 0.8  # Reduce size in choppy markets
        elif trend_strength > 0.5:
            return "trending", 0.6  # Unfavorable for mean-reversion
        else:
            return "neutral", 1.0
    
    def calculate_kelly_position_size(self):
        """
        Calculate optimal position size using Kelly Criterion.
        
        Kelly formula: f = (p × W - L) / W
        where p = win probability, W = average win, L = average loss
        
        We use quarter-Kelly for safety, which research shows is optimal
        for minimizing drawdowns while maintaining good returns.
        """
        if len(self.trade_history) < MIN_TRADES_FOR_KELLY:
            return BASE_RISK_PCT
        
        recent_trades = self.trade_history[-50:]  # Use last 50 trades
        wins = [t for t in recent_trades if t['pnl'] > 0]
        losses = [t for t in recent_trades if t['pnl'] < 0]
        
        if not wins or not losses:
            return BASE_RISK_PCT
        
        win_rate = len(wins) / len(recent_trades)
        avg_win = np.mean([t['pnl'] for t in wins])
        avg_loss = abs(np.mean([t['pnl'] for t in losses]))
        
        if avg_win == 0:
            return BASE_RISK_PCT
        
        # Kelly Criterion
        kelly = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
        kelly_risk = max(0.01, min(kelly * KELLY_FRACTION, MAX_RISK_PCT))
        
        return kelly_risk
    
    def check_daily_loss_limit(self, current_date):
        """
        Check if daily loss limit has been exceeded.
        This is a critical risk control - if we're having a bad day,
        we stop trading to prevent emotional decision-making and
        allow time to reassess market conditions.
        """
        date_key = current_date.date()
        daily_pnl = self.daily_pnl.get(date_key, 0.0)
        
        if daily_pnl < -CAPITAL * MAX_DAILY_LOSS_PCT:
            return True  # Stop trading for the day
        return False
    
    def identify_trap_patterns(self, df):
        """
        PHASE 2: Enhanced signal identification with quality filters.
        
        This is the core pattern recognition logic. We're looking for
        "liquidity traps" where price temporarily breaks through the
        Q1 range to trigger stops, then reverses back. This is a classic
        manipulation pattern used by larger market participants.
        
        Improvements over original:
        - Volume confirmation (real institutional activity vs. noise)
        - Time-of-day filtering (patterns behave differently throughout Q2)
        - Gap filtering (different behavior on gap days)
        - ATR-based stops (adapt to volatility)
        """
        df['time_only'] = df.index.time
        t_open = pd.Timestamp("08:30").time()
        t_q1_end = pd.Timestamp("10:00").time()
        t_q2_end = pd.Timestamp("11:30").time()
        
        signals = []
        grouped = df.groupby(df.index.date)
        
        for date, day_data in tqdm(grouped, desc="🔍 Scanning for High-Quality Trap Patterns"):
            if len(day_data) < 30:
                continue
            
            # Check if we've exceeded daily loss limit
            if self.check_daily_loss_limit(pd.Timestamp(date)):
                continue
            
            # Count trades today (avoid overtrading)
            daily_trades = len([s for s in signals if s['time'].date() == date])
            if daily_trades >= MAX_DAILY_TRADES:
                continue
            
            # ============================================================
            # Q1 RANGE ESTABLISHMENT (8:30 AM - 10:00 AM)
            # ============================================================
            q1_data = day_data[
                (day_data['time_only'] >= t_open) & 
                (day_data['time_only'] < t_q1_end)
            ]
            
            if len(q1_data) < MIN_Q1_BARS:
                continue
            
            q1_high = q1_data['high'].max()
            q1_low = q1_data['low'].min()
            q1_avg_volume = q1_data['volume'].mean()
            q1_close = q1_data.iloc[-1]['close']
            q1_open = q1_data.iloc[0]['open']
            
            # Gap detection (PHASE 2 filter)
            overnight_gap = abs(q1_open - day_data.iloc[0]['close']) if len(day_data) > len(q1_data) else 0
            gap_threshold = q1_data['atr'].iloc[-1] if not pd.isna(q1_data['atr'].iloc[-1]) else 0.5
            is_gap_day = overnight_gap > gap_threshold
            
            # ============================================================
            # Q2 TRAP PATTERN SCANNING (10:00 AM - 11:30 AM)
            # ============================================================
            q2_data = day_data[
                (day_data['time_only'] >= t_q1_end) & 
                (day_data['time_only'] < t_q2_end)
            ]
            
            if q2_data.empty:
                continue
            
            # Get current ATR for dynamic stops
            current_atr = q2_data['atr'].iloc[0]
            if pd.isna(current_atr):
                current_atr = (q1_high - q1_low) * 0.5  # Fallback to Q1 range
            
            candles = q2_data.rename_axis('timestamp').reset_index().to_dict('records')
            
            for i in range(len(candles)):
                curr = candles[i]
                prev = candles[i-1] if i > 0 else None
                
                signal = 0
                setup_type = ""
                stop_loss = 0
                confidence = 1.0
                
                # ===========================================================
                # PATTERN TYPE A: WICK SWEEP (Single Candle Rejection)
                # ===========================================================
                # Price pierces Q1 range but closes back inside - classic trap
                
                if curr['low'] < q1_low and curr['close'] > q1_low:
                    signal = 1  # Bullish reversal
                    setup_type = "Wick Sweep Long"
                    stop_loss = curr['low'] - (current_atr * ATR_STOP_MULTIPLIER)
                    
                    # Volume confirmation (PHASE 2)
                    if curr['volume'] < q1_avg_volume * MIN_VOLUME_MULTIPLIER:
                        confidence *= 0.7  # Reduce confidence on low volume
                    
                elif curr['high'] > q1_high and curr['close'] < q1_high:
                    signal = -1  # Bearish reversal
                    setup_type = "Wick Sweep Short"
                    stop_loss = curr['high'] + (current_atr * ATR_STOP_MULTIPLIER)
                    
                    if curr['volume'] < q1_avg_volume * MIN_VOLUME_MULTIPLIER:
                        confidence *= 0.7
                
                # ===========================================================
                # PATTERN TYPE B: FAKEOUT RECLAIM (Two Candle Reversal)
                # ===========================================================
                # Previous candle breaks Q1 range, current candle reclaims it
                
                if signal == 0 and prev:  # Only check if no wick sweep found
                    if prev['close'] < q1_low and curr['close'] > q1_low:
                        signal = 1
                        setup_type = "Fakeout Reclaim Long"
                        stop_loss = min(prev['low'], curr['low']) - (current_atr * ATR_STOP_MULTIPLIER)
                        
                        if curr['volume'] < q1_avg_volume * MIN_VOLUME_MULTIPLIER:
                            confidence *= 0.7
                    
                    elif prev['close'] > q1_high and curr['close'] < q1_high:
                        signal = -1
                        setup_type = "Fakeout Reclaim Short"
                        stop_loss = max(prev['high'], curr['high']) + (current_atr * ATR_STOP_MULTIPLIER)
                        
                        if curr['volume'] < q1_avg_volume * MIN_VOLUME_MULTIPLIER:
                            confidence *= 0.7
                
                # ===========================================================
                # SIGNAL QUALITY FILTERS (PHASE 2)
                # ===========================================================
                
                if signal != 0:
                    # Time-of-day adjustment
                    entry_time = curr['timestamp'].time()
                    if entry_time < pd.Timestamp("10:15").time():
                        confidence *= 1.1  # Early Q2 patterns more reliable
                    elif entry_time > pd.Timestamp("11:00").time():
                        confidence *= 0.9  # Late Q2 patterns less reliable
                    
                    # Gap day adjustment
                    if is_gap_day:
                        confidence *= 0.85  # Reduce confidence on gap days
                    
                    # Detect regime
                    regime, regime_multiplier = self.detect_market_regime(
                        df, curr['timestamp']
                    )
                    
                    signals.append({
                        'time': curr['timestamp'],
                        'signal': signal,
                        'entry': curr['close'],
                        'stop': stop_loss,
                        'type': setup_type,
                        'atr': current_atr,
                        'confidence': confidence,
                        'regime': regime,
                        'regime_multiplier': regime_multiplier,
                        'volume': curr['volume'],
                        'q1_high': q1_high,
                        'q1_low': q1_low
                    })
                    
                    # Priority to wick sweeps (more reliable)
                    if "Wick Sweep" in setup_type:
                        break
        
        return pd.DataFrame(signals)
    
    def execute_trade(self, setup, df, equity):
        """
        PHASE 3: Execute trade with dynamic risk management.
        
        This handles the full trade lifecycle from entry to exit,
        including:
        - Kelly Criterion position sizing
        - Regime-adjusted sizing
        - Dynamic stop loss management
        - MAE/MFE tracking for analysis
        """
        entry = setup['entry']
        stop = setup['stop']
        direction = setup['signal']
        setup_type = setup['type']
        confidence = setup['confidence']
        regime_multiplier = setup['regime_multiplier']
        
        # ============================================================
        # PHASE 3: DYNAMIC POSITION SIZING
        # ============================================================
        
        # Calculate base risk using Kelly Criterion
        kelly_risk = self.calculate_kelly_position_size()
        
        # Apply confidence and regime adjustments
        adjusted_risk = kelly_risk * confidence * regime_multiplier
        
        # Apply position scaling from consecutive losses
        adjusted_risk *= self.current_position_scale
        
        # Final risk amount in dollars
        risk_amount = equity * adjusted_risk
        
        # Calculate shares (fractional, then rounded)
        stop_distance = abs(entry - stop)
        shares = risk_amount / stop_distance
        
        # Apply minimum position size (PHASE 1 fix)
        if shares < MIN_POSITION_SIZE:
            shares = MIN_POSITION_SIZE
        else:
            shares = int(shares)
        
        # Calculate transaction costs (PHASE 1)
        transaction_cost = self.calculate_transaction_costs(shares, entry)
        
        # Calculate profit targets
        take_profit = entry + (stop_distance * TAKE_PROFIT_RATIO * direction)
        breakeven_trigger = entry + (stop_distance * BREAKEVEN_RATIO * direction)
        
        # ============================================================
        # TRADE EXECUTION SIMULATION
        # ============================================================
        
        future_data = df[df.index > setup['time']].head(48)  # Next 4 hours
        
        outcome = 0  # 0=time, 1=win, -1=loss
        stop_moved_to_breakeven = False
        exit_price = entry
        exit_reason = "time"
        
        # Track MAE/MFE (PHASE 4)
        max_favorable = 0
        max_adverse = 0
        
        for idx, bar in future_data.iterrows():
            current_stop = entry if stop_moved_to_breakeven else stop
            
            if direction == 1:  # Long position
                # Track excursions
                favorable_move = bar['high'] - entry
                adverse_move = entry - bar['low']
                max_favorable = max(max_favorable, favorable_move)
                max_adverse = max(max_adverse, adverse_move)
                
                # Check stop loss
                if bar['low'] <= current_stop:
                    exit_price = current_stop
                    outcome = -1 if not stop_moved_to_breakeven else 0
                    exit_reason = "stop_loss" if not stop_moved_to_breakeven else "breakeven"
                    break
                
                # Check take profit
                if bar['high'] >= take_profit:
                    exit_price = take_profit
                    outcome = 1
                    exit_reason = "take_profit"
                    break
                
                # Move stop to breakeven
                if bar['high'] >= breakeven_trigger and not stop_moved_to_breakeven:
                    stop_moved_to_breakeven = True
            
            else:  # Short position
                favorable_move = entry - bar['low']
                adverse_move = bar['high'] - entry
                max_favorable = max(max_favorable, favorable_move)
                max_adverse = max(max_adverse, adverse_move)
                
                if bar['high'] >= current_stop:
                    exit_price = current_stop
                    outcome = -1 if not stop_moved_to_breakeven else 0
                    exit_reason = "stop_loss" if not stop_moved_to_breakeven else "breakeven"
                    break
                
                if bar['low'] <= take_profit:
                    exit_price = take_profit
                    outcome = 1
                    exit_reason = "take_profit"
                    break
                
                if bar['low'] <= breakeven_trigger and not stop_moved_to_breakeven:
                    stop_moved_to_breakeven = True
        
        # Time-based exit if no other condition met
        if outcome == 0 and not future_data.empty:
            exit_price = future_data.iloc[-1]['close']
            # If stop moved to breakeven and we're losing, it's breakeven not loss
            if stop_moved_to_breakeven:
                if (direction == 1 and exit_price < entry) or (direction == -1 and exit_price > entry):
                    exit_price = entry
                    exit_reason = "breakeven"
        
        # ============================================================
        # PNL CALCULATION
        # ============================================================
        
        gross_pnl = (exit_price - entry) * shares * direction
        net_pnl = gross_pnl - transaction_cost
        
        # Track for Kelly Criterion
        trade_record = {
            'time': setup['time'],
            'type': setup_type,
            'entry': entry,
            'exit': exit_price,
            'shares': shares,
            'direction': direction,
            'pnl': net_pnl,
            'gross_pnl': gross_pnl,
            'transaction_cost': transaction_cost,
            'outcome': outcome,
            'exit_reason': exit_reason,
            'regime': setup['regime'],
            'confidence': confidence,
            'mae': max_adverse,
            'mfe': max_favorable,
            'risk_pct': adjusted_risk
        }
        
        self.trade_history.append(trade_record)
        
        # Update daily PnL tracking
        trade_date = setup['time'].date()
        self.daily_pnl[trade_date] = self.daily_pnl.get(trade_date, 0.0) + net_pnl
        
        # ============================================================
        # PHASE 3: DYNAMIC RISK SCALING
        # ============================================================
        
        if net_pnl < 0:
            self.consecutive_losses += 1
            if self.consecutive_losses >= MAX_CONSECUTIVE_LOSSES:
                self.current_position_scale = POSITION_SCALE_DOWN
                logger.warning(f"⚠️  Position size reduced to {POSITION_SCALE_DOWN*100:.0f}% after {self.consecutive_losses} consecutive losses")
        else:
            if self.consecutive_losses >= MAX_CONSECUTIVE_LOSSES:
                self.current_position_scale = 1.0
                logger.info(f"✅ Position size restored to 100% after winning trade")
            self.consecutive_losses = 0
        
        # Store MAE/MFE for analysis (PHASE 4)
        self.mae_mfe_data.append({
            'mae': max_adverse,
            'mfe': max_favorable,
            'outcome': outcome,
            'type': setup_type
        })
        
        return net_pnl, trade_record
    
    def monte_carlo_simulation(self):
        """
        PHASE 4: Monte Carlo simulation for confidence intervals.
        
        This resamples our actual trade outcomes to show the range of
        possible equity curves we might experience. It answers the question:
        "How much of our performance is luck vs. skill?"
        
        Returns distribution of final equity values and maximum drawdowns.
        """
        if len(self.trade_history) < 30:
            return None
        
        logger.info(f"🎲 Running Monte Carlo Simulation ({MONTE_CARLO_RUNS:,} iterations)...")
        
        trade_pnls = [t['pnl'] for t in self.trade_history]
        
        final_equities = []
        max_drawdowns = []
        
        for _ in tqdm(range(MONTE_CARLO_RUNS), desc="Monte Carlo"):
            # Resample trades with replacement
            simulated_pnls = np.random.choice(trade_pnls, size=len(trade_pnls), replace=True)
            
            # Calculate equity curve
            equity_curve = CAPITAL + np.cumsum(simulated_pnls)
            final_equity = equity_curve[-1]
            
            # Calculate max drawdown
            running_max = np.maximum.accumulate(equity_curve)
            drawdown = (equity_curve - running_max) / running_max
            max_dd = abs(drawdown.min())
            
            final_equities.append(final_equity)
            max_drawdowns.append(max_dd)
        
        results = {
            'final_equity': {
                'mean': np.mean(final_equities),
                'median': np.median(final_equities),
                'std': np.std(final_equities),
                'percentile_5': np.percentile(final_equities, 5),
                'percentile_25': np.percentile(final_equities, 25),
                'percentile_75': np.percentile(final_equities, 75),
                'percentile_95': np.percentile(final_equities, 95),
            },
            'max_drawdown': {
                'mean': np.mean(max_drawdowns),
                'median': np.median(max_drawdowns),
                'percentile_95': np.percentile(max_drawdowns, 95)
            }
        }
        
        return results
    
    def analyze_mae_mfe(self):
        """
        PHASE 4: Maximum Adverse/Favorable Excursion Analysis.
        
        MAE shows us how far trades moved against us before outcome.
        MFE shows us how far in our favor they moved.
        
        This reveals whether our stops are optimal or if we're:
        - Stopping out too early (high MAE on winners)
        - Giving back too much profit (high MFE on losers)
        """
        if not self.mae_mfe_data:
            return None
        
        df = pd.DataFrame(self.mae_mfe_data)
        
        winners = df[df['outcome'] == 1]
        losers = df[df['outcome'] == -1]
        
        analysis = {
            'winners': {
                'avg_mae': winners['mae'].mean() if len(winners) > 0 else 0,
                'avg_mfe': winners['mfe'].mean() if len(winners) > 0 else 0,
                'count': len(winners)
            },
            'losers': {
                'avg_mae': losers['mae'].mean() if len(losers) > 0 else 0,
                'avg_mfe': losers['mfe'].mean() if len(losers) > 0 else 0,
                'count': len(losers)
            }
        }
        
        return analysis
    
    def generate_performance_report(self, equity_curve, signals_df):
        """
        Generate comprehensive performance statistics for the report.
        """
        if not self.trade_history:
            return None
        
        # Calculate returns
        total_return = (equity_curve[-1] - CAPITAL) / CAPITAL
        
        # Win rate analysis
        winners = [t for t in self.trade_history if t['pnl'] > 0]
        losers = [t for t in self.trade_history if t['pnl'] < 0]
        win_rate = len(winners) / len(self.trade_history) if self.trade_history else 0
        
        # Average win/loss
        avg_win = np.mean([t['pnl'] for t in winners]) if winners else 0
        avg_loss = np.mean([t['pnl'] for t in losers]) if losers else 0
        profit_factor = abs(sum([t['pnl'] for t in winners]) / sum([t['pnl'] for t in losers])) if losers else float('inf')
        
        # Drawdown analysis
        running_max = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - running_max) / running_max
        max_drawdown = abs(drawdown.min())
        
        # Sharpe ratio (annualized)
        returns = pd.Series(equity_curve).pct_change().dropna()
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252 * 78) if returns.std() > 0 else 0  # 78 trading periods per day
        
        # Strategy breakdown
        strategy_stats = {}
        for stype in signals_df['type'].unique():
            trades = [t for t in self.trade_history if t['type'] == stype]
            if trades:
                wins = [t for t in trades if t['pnl'] > 0]
                strategy_stats[stype] = {
                    'count': len(trades),
                    'win_rate': len(wins) / len(trades),
                    'total_pnl': sum([t['pnl'] for t in trades]),
                    'avg_pnl': np.mean([t['pnl'] for t in trades])
                }
        
        # Regime analysis
        regime_stats = {}
        for regime in ['mean_reverting', 'high_volatility', 'trending', 'neutral']:
            trades = [t for t in self.trade_history if t['regime'] == regime]
            if trades:
                wins = [t for t in trades if t['pnl'] > 0]
                regime_stats[regime] = {
                    'count': len(trades),
                    'win_rate': len(wins) / len(trades),
                    'total_pnl': sum([t['pnl'] for t in trades])
                }
        
        report = {
            'capital': {
                'starting': CAPITAL,
                'ending': equity_curve[-1],
                'total_return': total_return,
                'total_pnl': equity_curve[-1] - CAPITAL
            },
            'trades': {
                'total': len(self.trade_history),
                'winners': len(winners),
                'losers': len(losers),
                'breakeven': len(self.trade_history) - len(winners) - len(losers)
            },
            'performance': {
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': profit_factor,
                'expectancy': win_rate * avg_win + (1 - win_rate) * avg_loss
            },
            'risk': {
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe,
                'avg_transaction_cost': np.mean([t['transaction_cost'] for t in self.trade_history])
            },
            'strategy_breakdown': strategy_stats,
            'regime_breakdown': regime_stats
        }
        
        return report
    
    def run(self):
        """
        Main backtest execution with all optimization phases.
        """
        logger.info("=" * 70)
        logger.info("🚀 MOKSHA CAPITAL - OPTIMIZED BACKTEST ENGINE v2.0")
        logger.info("=" * 70)
        
        # Fetch data
        df = self.fetch_real_data()
        if df.empty:
            logger.error("❌ No data available for backtest")
            return None
        
        # Identify signals with quality filters (PHASE 2)
        signals = self.identify_trap_patterns(df)
        
        if signals.empty:
            logger.error("❌ No valid signals found")
            return None
        
        logger.info(f"✅ Found {len(signals)} high-quality trap patterns")
        
        # Execute trades with dynamic risk management (PHASE 3)
        equity = CAPITAL
        equity_curve = [CAPITAL]
        
        logger.info("\n" + "=" * 70)
        logger.info("💼 EXECUTING TRADES WITH DYNAMIC RISK MANAGEMENT")
        logger.info("=" * 70)
        
        for idx, setup in tqdm(signals.iterrows(), total=len(signals), desc="⚡ Processing Trades"):
            pnl, trade_record = self.execute_trade(setup, df, equity)
            equity += pnl
            equity_curve.append(equity)
        
        # Generate performance report
        performance = self.generate_performance_report(equity_curve, signals)
        
        # Monte Carlo simulation (PHASE 4)
        mc_results = self.monte_carlo_simulation()
        
        # MAE/MFE analysis (PHASE 4)
        mae_mfe = self.analyze_mae_mfe()
        
        # ============================================================
        # RESULTS OUTPUT
        # ============================================================
        
        logger.info("\n" + "=" * 70)
        logger.info("📊 BACKTEST RESULTS - MOKSHA CAPITAL MANAGEMENT")
        logger.info("=" * 70)
        
        logger.info(f"\n💰 CAPITAL PERFORMANCE")
        logger.info(f"   Starting Capital:     ${performance['capital']['starting']:,.2f}")
        logger.info(f"   Ending Capital:       ${performance['capital']['ending']:,.2f}")
        logger.info(f"   Total Return:         {performance['capital']['total_return']*100:.2f}%")
        logger.info(f"   Net Profit:           ${performance['capital']['total_pnl']:,.2f}")
        
        logger.info(f"\n📈 TRADE STATISTICS")
        logger.info(f"   Total Trades:         {performance['trades']['total']}")
        logger.info(f"   Winners:              {performance['trades']['winners']}")
        logger.info(f"   Losers:               {performance['trades']['losers']}")
        logger.info(f"   Win Rate:             {performance['performance']['win_rate']*100:.1f}%")
        logger.info(f"   Profit Factor:        {performance['performance']['profit_factor']:.2f}")
        logger.info(f"   Expectancy:           ${performance['performance']['expectancy']:.2f}")
        
        logger.info(f"\n💵 WIN/LOSS ANALYSIS")
        logger.info(f"   Average Win:          ${performance['performance']['avg_win']:.2f}")
        logger.info(f"   Average Loss:         ${performance['performance']['avg_loss']:.2f}")
        logger.info(f"   Avg Transaction Cost: ${performance['risk']['avg_transaction_cost']:.2f}")
        
        logger.info(f"\n⚠️  RISK METRICS")
        logger.info(f"   Maximum Drawdown:     {performance['risk']['max_drawdown']*100:.2f}%")
        logger.info(f"   Sharpe Ratio:         {performance['risk']['sharpe_ratio']:.2f}")
        
        logger.info(f"\n🎯 STRATEGY BREAKDOWN")
        for strategy, stats in performance['strategy_breakdown'].items():
            logger.info(f"   {strategy}:")
            logger.info(f"      Trades: {stats['count']} | Win Rate: {stats['win_rate']*100:.1f}% | PnL: ${stats['total_pnl']:,.2f}")
        
        logger.info(f"\n🌡️  REGIME ANALYSIS")
        for regime, stats in performance['regime_breakdown'].items():
            logger.info(f"   {regime.replace('_', ' ').title()}:")
            logger.info(f"      Trades: {stats['count']} | Win Rate: {stats['win_rate']*100:.1f}% | PnL: ${stats['total_pnl']:,.2f}")
        
        if mc_results:
            logger.info(f"\n🎲 MONTE CARLO SIMULATION ({MONTE_CARLO_RUNS:,} iterations)")
            logger.info(f"   Expected Final Equity:  ${mc_results['final_equity']['median']:,.2f}")
            logger.info(f"   5th Percentile:         ${mc_results['final_equity']['percentile_5']:,.2f}")
            logger.info(f"   95th Percentile:        ${mc_results['final_equity']['percentile_95']:,.2f}")
            logger.info(f"   Expected Max Drawdown:  {mc_results['max_drawdown']['median']*100:.2f}%")
            logger.info(f"   95th Percentile DD:     {mc_results['max_drawdown']['percentile_95']*100:.2f}%")
        
        if mae_mfe:
            logger.info(f"\n📉 MAE/MFE ANALYSIS")
            logger.info(f"   Winners - Avg MAE: ${mae_mfe['winners']['avg_mae']:.2f} | Avg MFE: ${mae_mfe['winners']['avg_mfe']:.2f}")
            logger.info(f"   Losers  - Avg MAE: ${mae_mfe['losers']['avg_mae']:.2f} | Avg MFE: ${mae_mfe['losers']['avg_mfe']:.2f}")
        
        logger.info("\n" + "=" * 70)
        logger.info("✅ BACKTEST COMPLETE")
        logger.info("=" * 70)
        
        # Return comprehensive results for report generation
        return {
            'performance': performance,
            'monte_carlo': mc_results,
            'mae_mfe': mae_mfe,
            'equity_curve': equity_curve,
            'trade_history': self.trade_history,
            'signals': signals
        }


# ============================================================================
# EXECUTION
# ============================================================================

if __name__ == "__main__":
    backtest = MokshaOptimizedBacktest()
    results = backtest.run()
    
    if results:
        # Save detailed results to JSON for further analysis
        output_file = f"/Users/viveksingh/Documents/Patience/Analysis/TARS/MOKSHA/Moksha_1/moksha_backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Convert non-serializable objects
        serializable_results = {
            'performance': results['performance'],
            'monte_carlo': results['monte_carlo'],
            'mae_mfe': results['mae_mfe'],
            'equity_curve': results['equity_curve'],
            'total_trades': len(results['trade_history']),
            'config': {
                'symbol': SYMBOL,
                'capital': CAPITAL,
                'base_risk': BASE_RISK_PCT,
                'atr_multiplier': ATR_STOP_MULTIPLIER,
                'kelly_fraction': KELLY_FRACTION
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"\n💾 Results saved to: {output_file}")