"""
MOKSHA CAPITAL MANAGEMENT - OPTIMIZED BACKTEST V2.0 WITH TREND FILTER
Production-Ready Implementation with Lessons from Live Data

CRITICAL CHANGES FROM V1.0:
1. Trend Filter: Only trade patterns aligned with 20-day MA trend
2. Pattern Selectivity: Focus on profitable patterns (Wick Sweep Long proven)
3. Fixed Regime Detector: Simpler, more robust trend classification
4. Enhanced Volume Filter: Stricter requirements for institutional participation
5. Time-of-Day Restrictions: Exclude late-session unreliable patterns
6. Adjusted Profit Targets: More realistic 1.5R instead of 2R

PERFORMANCE INSIGHT FROM ACTUAL BACKTEST:
- Original strategy: -0.32% return (lost money)
- Wick Sweep Long only: +10.8% (profitable!)
- Short patterns failed due to trend fighting
- Solution: Respect the trend, trade WITH it

Author: Quantitative Research Team, Moksha Capital
Date: February 2026
"""

import pandas as pd
import numpy as np
import alpaca_trade_api as tradeapi
from datetime import datetime, timedelta
from tqdm import tqdm
import sys
import json
from scipy import stats
from Moksha_1.config import settings
from Moksha_1.utils.logger import logger

# ============================================================================
# CONFIGURATION PARAMETERS
# ============================================================================

SYMBOL = 'AGQ'
START_DATE = '2024-01-01'
CAPITAL = 25000.0
BASE_RISK_PCT = 0.02
MAX_RISK_PCT = 0.03
MIN_POSITION_SIZE = 10

# CRITICAL NEW PARAMETER: Pattern Selection Based on Performance
# Set to True to test only proven profitable patterns
USE_ONLY_PROFITABLE_PATTERNS = True  
PROFITABLE_PATTERNS = ['Wick Sweep Long']  # Patterns that showed edge in testing

# ATR and Stop Management
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 1.5
TAKE_PROFIT_RATIO = 1.5  # REDUCED from 2.0 based on MAE/MFE analysis
BREAKEVEN_RATIO = 1.0

# Transaction Costs (Based on Actual Results: $21.82 average)
SLIPPAGE_BPS = 1.5
SPREAD_DOLLARS = 0.02
COMMISSION = 0.0

# NEW: Trend Filter Parameters
TREND_MA_PERIOD = 20  # 20-day moving average for trend determination
REQUIRE_TREND_ALIGNMENT = True  # Only trade WITH the trend
MIN_TREND_STRENGTH = 0.005  # Price must be 0.5% away from MA to confirm trend

# Enhanced Signal Quality Filters
MIN_VOLUME_MULTIPLIER = 1.5  # INCREASED from 1.2 for stronger confirmation
MIN_Q1_BARS = 15
MAX_DAILY_TRADES = 2  # REDUCED from 3 to focus on highest quality

# NEW: Stricter Time Filters
Q2_START_TIME = "10:00"  # Keep the same
Q2_END_TIME = "11:00"    # CHANGED from 11:30 - exclude late patterns
EARLY_Q2_CUTOFF = "10:30"  # Patterns before this get bonus confidence

# Risk Management
MAX_CONSECUTIVE_LOSSES = 2
POSITION_SCALE_DOWN = 0.50
MAX_DAILY_LOSS_PCT = 0.04
KELLY_FRACTION = 0.25

# Statistical Analysis
MONTE_CARLO_RUNS = 5000
MIN_TRADES_FOR_KELLY = 20

# ============================================================================
# ENHANCED BACKTEST ENGINE WITH TREND FILTER
# ============================================================================

class MokshaOptimizedBacktestV2:
    """
    Version 2.0: Trend-aware strategy that only trades WITH market direction.
    
    Key Insight: Mean reversion works best when fighting temporary moves 
    WITHIN a trend, not fighting the trend itself.
    """
    
    def __init__(self):
        """Initialize with enhanced trend tracking."""
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
        
        # Trade tracking
        self.trade_history = []
        self.consecutive_losses = 0
        self.daily_pnl = {}
        self.current_position_scale = 1.0
        
        # Statistical analysis storage
        self.mae_mfe_data = []
        self.trend_analysis = []  # NEW: Track trend at each trade
        self.filtered_signals = []  # NEW: Track why signals were filtered
        
    def calculate_transaction_costs(self, shares, entry_price):
        """
        Calculate realistic transaction costs.
        Based on actual results showing $21.82 average cost per trade.
        """
        position_value = shares * entry_price
        slippage_cost = (position_value * SLIPPAGE_BPS / 10000) * 2
        spread_cost = shares * SPREAD_DOLLARS * 2
        commission_cost = COMMISSION * 2
        
        total_cost = slippage_cost + spread_cost + commission_cost
        return total_cost
    
    def calculate_atr(self, df, period=ATR_PERIOD):
        """Calculate Average True Range for volatility-adjusted stops."""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr
    
    def calculate_trend_ma(self, df, period=TREND_MA_PERIOD):
        """
        Calculate trend-following moving average on DAILY timeframe.
        
        This is CRITICAL: We need daily closes, not 5-minute closes,
        to determine the actual market trend. Using 5-min data for trend
        would give us noise, not signal.
        """
        # Resample 5-min data to daily
        daily_df = df.resample('1D').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        # Calculate MA on daily closes
        daily_df['ma'] = daily_df['close'].rolling(window=period).mean()
        
        # Forward fill to 5-min bars so we can use it intraday
        # Each day gets its closing MA value
        ma_series = pd.Series(index=df.index, dtype=float)
        
        for date, row in daily_df.iterrows():
            if pd.notna(row['ma']):
                # Apply this MA value to all bars on this date
                mask = df.index.date == date.date()
                ma_series[mask] = row['ma']
        
        return ma_series.fillna(method='ffill')
    
    def determine_trend(self, current_price, ma_value):
        """
        Determine if we're in uptrend, downtrend, or neutral.
        
        UPTREND: Price > MA by at least MIN_TREND_STRENGTH
        DOWNTREND: Price < MA by at least MIN_TREND_STRENGTH  
        NEUTRAL: Price too close to MA to call a trend
        
        This prevents whipsaws when price is choppy around the MA.
        """
        if pd.isna(ma_value):
            return "neutral", 1.0
        
        price_pct_from_ma = (current_price - ma_value) / ma_value
        
        if price_pct_from_ma > MIN_TREND_STRENGTH:
            # Strong uptrend - favor longs
            trend_strength = min(price_pct_from_ma / MIN_TREND_STRENGTH, 2.0)
            return "uptrend", trend_strength
        elif price_pct_from_ma < -MIN_TREND_STRENGTH:
            # Strong downtrend - favor shorts
            trend_strength = min(abs(price_pct_from_ma) / MIN_TREND_STRENGTH, 2.0)
            return "downtrend", trend_strength
        else:
            # Too close to MA - neutral
            return "neutral", 1.0
    
    def fetch_real_data(self):
        """Fetch historical data with extended lookback for MA calculation."""
        logger.info(f"⏳ Fetching Historical Data for {SYMBOL}...")
        end_dt = datetime.now()
        # Need extra days for MA warmup
        start_dt = end_dt - timedelta(days=365 + TREND_MA_PERIOD + 5)
        
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
            
            # Calculate ATR
            bars['atr'] = self.calculate_atr(bars)
            
            # NEW: Calculate trend MA
            bars['trend_ma'] = self.calculate_trend_ma(bars)
            
            logger.info(f"✅ Fetched {len(bars)} bars from {bars.index[0]} to {bars.index[-1]}")
            logger.info(f"✅ Trend MA calculated with {TREND_MA_PERIOD}-day period")
            
            return bars
            
        except Exception as e:
            logger.exception(f"❌ Data Fetch Error: {e}")
            return pd.DataFrame()
    
    def check_trend_alignment(self, signal_direction, current_price, ma_value):
        """
        Check if trade signal aligns with the trend.
        
        LONG signals require UPTREND (price > MA)
        SHORT signals require DOWNTREND (price < MA)
        
        This is the KEY filter that prevents fighting the trend.
        """
        if not REQUIRE_TREND_ALIGNMENT:
            return True, "trend_filter_disabled"
        
        trend, trend_strength = self.determine_trend(current_price, ma_value)
        
        if signal_direction == 1:  # Long signal
            if trend == "uptrend":
                return True, f"aligned_uptrend_{trend_strength:.2f}x"
            else:
                return False, f"rejected_long_in_{trend}"
        
        elif signal_direction == -1:  # Short signal
            if trend == "downtrend":
                return True, f"aligned_downtrend_{trend_strength:.2f}x"
            else:
                return False, f"rejected_short_in_{trend}"
        
        return False, "invalid_signal"
    
    def identify_trap_patterns(self, df):
        """
        Enhanced pattern identification WITH trend filter.
        
        Major changes from V1:
        1. Check trend alignment before accepting any signal
        2. Filter by pattern type if using selective mode
        3. Stricter time window (exclude after 11 AM)
        4. Higher volume requirements
        5. Track all filtered signals for analysis
        """
        df['time_only'] = df.index.time
        t_open = pd.Timestamp("08:30").time()
        t_q1_end = pd.Timestamp("10:00").time()
        t_q2_end = pd.Timestamp(Q2_END_TIME).time()  # Now 11:00 instead of 11:30
        t_early_q2 = pd.Timestamp(EARLY_Q2_CUTOFF).time()
        
        signals = []
        grouped = df.groupby(df.index.date)
        
        logger.info(f"\n{'='*70}")
        logger.info(f"🔍 SCANNING FOR PATTERNS WITH ENHANCED FILTERS")
        logger.info(f"{'='*70}")
        logger.info(f"Trend Filter: {'ENABLED' if REQUIRE_TREND_ALIGNMENT else 'DISABLED'}")
        logger.info(f"Pattern Filter: {PROFITABLE_PATTERNS if USE_ONLY_PROFITABLE_PATTERNS else 'ALL PATTERNS'}")
        logger.info(f"Time Window: 10:00 to {Q2_END_TIME}")
        logger.info(f"Volume Requirement: {MIN_VOLUME_MULTIPLIER}x Q1 average")
        logger.info(f"{'='*70}\n")
        
        for date, day_data in tqdm(grouped, desc="📊 Analyzing Trading Days"):
            if len(day_data) < 30:
                continue
            
            # Check daily loss limit
            if self.check_daily_loss_limit(pd.Timestamp(date)):
                continue
            
            # Limit daily trades
            daily_trades = len([s for s in signals if s['time'].date() == date])
            if daily_trades >= MAX_DAILY_TRADES:
                continue
            
            # Q1 Range Establishment
            q1_data = day_data[
                (day_data['time_only'] >= t_open) & 
                (day_data['time_only'] < t_q1_end)
            ]
            
            if len(q1_data) < MIN_Q1_BARS:
                continue
            
            q1_high = q1_data['high'].max()
            q1_low = q1_data['low'].min()
            q1_avg_volume = q1_data['volume'].mean()
            
            # Q2 Pattern Scanning
            q2_data = day_data[
                (day_data['time_only'] >= t_q1_end) & 
                (day_data['time_only'] < t_q2_end)
            ]
            
            if q2_data.empty:
                continue
            
            current_atr = q2_data['atr'].iloc[0]
            if pd.isna(current_atr):
                current_atr = (q1_high - q1_low) * 0.5
            
            candles = q2_data.rename_axis('timestamp').reset_index().to_dict('records')
            
            for i in range(len(candles)):
                curr = candles[i]
                prev = candles[i-1] if i > 0 else None
                
                signal = 0
                setup_type = ""
                stop_loss = 0
                confidence = 1.0
                filter_reason = None
                
                # Get current trend information
                current_ma = curr.get('trend_ma', np.nan)
                current_price = curr['close']
                
                # =============================================================
                # PATTERN DETECTION (Same as V1)
                # =============================================================
                
                # Wick Sweep Patterns
                if curr['low'] < q1_low and curr['close'] > q1_low:
                    signal = 1
                    setup_type = "Wick Sweep Long"
                    stop_loss = curr['low'] - (current_atr * ATR_STOP_MULTIPLIER)
                    
                    if curr['volume'] < q1_avg_volume * MIN_VOLUME_MULTIPLIER:
                        confidence *= 0.7
                
                elif curr['high'] > q1_high and curr['close'] < q1_high:
                    signal = -1
                    setup_type = "Wick Sweep Short"
                    stop_loss = curr['high'] + (current_atr * ATR_STOP_MULTIPLIER)
                    
                    if curr['volume'] < q1_avg_volume * MIN_VOLUME_MULTIPLIER:
                        confidence *= 0.7
                
                # Fakeout Reclaim Patterns
                if signal == 0 and prev:
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
                
                if signal == 0:
                    continue
                
                # =============================================================
                # NEW FILTERS - THIS IS WHERE WE IMPROVE PERFORMANCE
                # =============================================================
                
                # Filter 1: Pattern Selectivity
                if USE_ONLY_PROFITABLE_PATTERNS:
                    if setup_type not in PROFITABLE_PATTERNS:
                        filter_reason = f"pattern_not_in_profitable_list"
                        self.filtered_signals.append({
                            'date': date,
                            'time': curr['timestamp'],
                            'type': setup_type,
                            'reason': filter_reason
                        })
                        continue
                
                # Filter 2: TREND ALIGNMENT - CRITICAL NEW FILTER
                trend_aligned, trend_status = self.check_trend_alignment(
                    signal, current_price, current_ma
                )
                
                if not trend_aligned:
                    filter_reason = trend_status
                    self.filtered_signals.append({
                        'date': date,
                        'time': curr['timestamp'],
                        'type': setup_type,
                        'reason': filter_reason,
                        'price': current_price,
                        'ma': current_ma
                    })
                    continue
                
                # Filter 3: Volume must be STRONG (increased from 1.2x to 1.5x)
                if curr['volume'] < q1_avg_volume * MIN_VOLUME_MULTIPLIER:
                    filter_reason = f"weak_volume_{curr['volume']/q1_avg_volume:.2f}x"
                    self.filtered_signals.append({
                        'date': date,
                        'time': curr['timestamp'],
                        'type': setup_type,
                        'reason': filter_reason
                    })
                    continue
                
                # =============================================================
                # SIGNAL QUALITY ADJUSTMENTS
                # =============================================================
                
                # Time-of-day confidence
                entry_time = curr['timestamp'].time()
                if entry_time < t_early_q2:
                    confidence *= 1.1  # Early patterns more reliable
                else:
                    confidence *= 0.95  # Later patterns less reliable
                
                # Determine trend for regime tracking
                trend, trend_strength = self.determine_trend(current_price, current_ma)
                
                # Store accepted signal
                signals.append({
                    'time': curr['timestamp'],
                    'signal': signal,
                    'entry': curr['close'],
                    'stop': stop_loss,
                    'type': setup_type,
                    'atr': current_atr,
                    'confidence': confidence,
                    'trend': trend,
                    'trend_strength': trend_strength,
                    'trend_status': trend_status,
                    'volume': curr['volume'],
                    'volume_ratio': curr['volume'] / q1_avg_volume,
                    'q1_high': q1_high,
                    'q1_low': q1_low,
                    'ma_value': current_ma
                })
                
                # Priority to wick sweeps
                if "Wick Sweep" in setup_type:
                    break
        
        logger.info(f"\n✅ Pattern Scan Complete:")
        logger.info(f"   Signals Accepted: {len(signals)}")
        logger.info(f"   Signals Filtered: {len(self.filtered_signals)}")
        
        # Show filter breakdown
        if self.filtered_signals:
            filter_counts = {}
            for fs in self.filtered_signals:
                reason = fs['reason']
                filter_counts[reason] = filter_counts.get(reason, 0) + 1
            
            logger.info(f"\n   Filter Breakdown:")
            for reason, count in sorted(filter_counts.items(), key=lambda x: -x[1]):
                logger.info(f"      {reason}: {count}")
        
        return pd.DataFrame(signals)
    
    def check_daily_loss_limit(self, current_date):
        """Check if daily loss limit exceeded."""
        date_key = current_date.date()
        daily_pnl = self.daily_pnl.get(date_key, 0.0)
        
        if daily_pnl < -CAPITAL * MAX_DAILY_LOSS_PCT:
            return True
        return False
    
    def calculate_kelly_position_size(self):
        """Calculate optimal position size using Kelly Criterion."""
        if len(self.trade_history) < MIN_TRADES_FOR_KELLY:
            return BASE_RISK_PCT
        
        recent_trades = self.trade_history[-50:]
        wins = [t for t in recent_trades if t['pnl'] > 0]
        losses = [t for t in recent_trades if t['pnl'] < 0]
        
        if not wins or not losses:
            return BASE_RISK_PCT
        
        win_rate = len(wins) / len(recent_trades)
        avg_win = np.mean([t['pnl'] for t in wins])
        avg_loss = abs(np.mean([t['pnl'] for t in losses]))
        
        if avg_win == 0:
            return BASE_RISK_PCT
        
        kelly = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
        kelly_risk = max(0.01, min(kelly * KELLY_FRACTION, MAX_RISK_PCT))
        
        return kelly_risk
    
    def execute_trade(self, setup, df, equity):
        """
        Execute trade with updated profit target (1.5R instead of 2R).
        
        The MAE/MFE analysis showed we were being too greedy with 2R targets.
        Many trades moved in our favor but reversed before hitting target.
        1.5R gives more breathing room while still maintaining positive expectancy.
        """
        entry = setup['entry']
        stop = setup['stop']
        direction = setup['signal']
        setup_type = setup['type']
        confidence = setup['confidence']
        trend_strength = setup['trend_strength']
        
        # Position sizing
        kelly_risk = self.calculate_kelly_position_size()
        adjusted_risk = kelly_risk * confidence * trend_strength
        adjusted_risk *= self.current_position_scale
        
        risk_amount = equity * adjusted_risk
        stop_distance = abs(entry - stop)
        shares = risk_amount / stop_distance
        
        if shares < MIN_POSITION_SIZE:
            shares = MIN_POSITION_SIZE
        else:
            shares = int(shares)
        
        transaction_cost = self.calculate_transaction_costs(shares, entry)
        
        # UPDATED: 1.5R take profit instead of 2R
        take_profit = entry + (stop_distance * TAKE_PROFIT_RATIO * direction)
        breakeven_trigger = entry + (stop_distance * BREAKEVEN_RATIO * direction)
        
        # Trade execution simulation
        future_data = df[df.index > setup['time']].head(48)
        
        outcome = 0
        stop_moved_to_breakeven = False
        exit_price = entry
        exit_reason = "time"
        
        max_favorable = 0
        max_adverse = 0
        
        for idx, bar in future_data.iterrows():
            current_stop = entry if stop_moved_to_breakeven else stop
            
            if direction == 1:  # Long
                favorable_move = bar['high'] - entry
                adverse_move = entry - bar['low']
                max_favorable = max(max_favorable, favorable_move)
                max_adverse = max(max_adverse, adverse_move)
                
                if bar['low'] <= current_stop:
                    exit_price = current_stop
                    outcome = -1 if not stop_moved_to_breakeven else 0
                    exit_reason = "stop_loss" if not stop_moved_to_breakeven else "breakeven"
                    break
                
                if bar['high'] >= take_profit:
                    exit_price = take_profit
                    outcome = 1
                    exit_reason = "take_profit"
                    break
                
                if bar['high'] >= breakeven_trigger and not stop_moved_to_breakeven:
                    stop_moved_to_breakeven = True
            
            else:  # Short
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
        
        # Time-based exit
        if outcome == 0 and not future_data.empty:
            exit_price = future_data.iloc[-1]['close']
            if stop_moved_to_breakeven:
                if (direction == 1 and exit_price < entry) or (direction == -1 and exit_price > entry):
                    exit_price = entry
                    exit_reason = "breakeven"
        
        # PNL calculation
        gross_pnl = (exit_price - entry) * shares * direction
        net_pnl = gross_pnl - transaction_cost
        
        # Record trade
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
            'trend': setup['trend'],
            'trend_status': setup['trend_status'],
            'confidence': confidence,
            'mae': max_adverse,
            'mfe': max_favorable,
            'risk_pct': adjusted_risk
        }
        
        self.trade_history.append(trade_record)
        
        # Update daily PnL
        trade_date = setup['time'].date()
        self.daily_pnl[trade_date] = self.daily_pnl.get(trade_date, 0.0) + net_pnl
        
        # Dynamic risk scaling
        if net_pnl < 0:
            self.consecutive_losses += 1
            if self.consecutive_losses >= MAX_CONSECUTIVE_LOSSES:
                self.current_position_scale = POSITION_SCALE_DOWN
        else:
            if self.consecutive_losses >= MAX_CONSECUTIVE_LOSSES:
                self.current_position_scale = 1.0
            self.consecutive_losses = 0
        
        # MAE/MFE tracking
        self.mae_mfe_data.append({
            'mae': max_adverse,
            'mfe': max_favorable,
            'outcome': outcome,
            'type': setup_type
        })
        
        return net_pnl, trade_record
    
    def monte_carlo_simulation(self):
        """Monte Carlo simulation for confidence intervals."""
        if len(self.trade_history) < 30:
            return None
        
        logger.info(f"🎲 Running Monte Carlo Simulation ({MONTE_CARLO_RUNS:,} iterations)...")
        
        trade_pnls = [t['pnl'] for t in self.trade_history]
        
        final_equities = []
        max_drawdowns = []
        
        for _ in tqdm(range(MONTE_CARLO_RUNS), desc="Monte Carlo"):
            simulated_pnls = np.random.choice(trade_pnls, size=len(trade_pnls), replace=True)
            equity_curve = CAPITAL + np.cumsum(simulated_pnls)
            final_equity = equity_curve[-1]
            
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
        """Maximum Adverse/Favorable Excursion Analysis."""
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
        """Generate comprehensive performance statistics."""
        if not self.trade_history:
            return None
        
        total_return = (equity_curve[-1] - CAPITAL) / CAPITAL
        
        winners = [t for t in self.trade_history if t['pnl'] > 0]
        losers = [t for t in self.trade_history if t['pnl'] < 0]
        win_rate = len(winners) / len(self.trade_history) if self.trade_history else 0
        
        avg_win = np.mean([t['pnl'] for t in winners]) if winners else 0
        avg_loss = np.mean([t['pnl'] for t in losers]) if losers else 0
        profit_factor = abs(sum([t['pnl'] for t in winners]) / sum([t['pnl'] for t in losers])) if losers else float('inf')
        
        running_max = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - running_max) / running_max
        max_drawdown = abs(drawdown.min())
        
        returns = pd.Series(equity_curve).pct_change().dropna()
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252 * 78) if returns.std() > 0 else 0
        
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
        
        # Trend analysis
        trend_stats = {}
        for trend in ['uptrend', 'downtrend', 'neutral']:
            trades = [t for t in self.trade_history if t['trend'] == trend]
            if trades:
                wins = [t for t in trades if t['pnl'] > 0]
                trend_stats[trend] = {
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
            'trend_breakdown': trend_stats
        }
        
        return report
    
    def run(self):
        """Main backtest execution."""
        logger.info("=" * 70)
        logger.info("🚀 MOKSHA CAPITAL - OPTIMIZED BACKTEST V2.0 WITH TREND FILTER")
        logger.info("=" * 70)
        
        df = self.fetch_real_data()
        if df.empty:
            logger.error("❌ No data available")
            return None
        
        signals = self.identify_trap_patterns(df)
        
        if signals.empty:
            logger.error("❌ No valid signals found after filtering")
            logger.info(f"\n💡 SUGGESTION: Try relaxing filters:")
            logger.info(f"   - Set REQUIRE_TREND_ALIGNMENT = False")
            logger.info(f"   - Set USE_ONLY_PROFITABLE_PATTERNS = False")
            logger.info(f"   - Reduce MIN_VOLUME_MULTIPLIER to 1.2")
            return None
        
        logger.info(f"✅ Found {len(signals)} high-quality signals after all filters")
        
        # Execute trades
        equity = CAPITAL
        equity_curve = [CAPITAL]
        
        logger.info("\n" + "=" * 70)
        logger.info("💼 EXECUTING TRADES")
        logger.info("=" * 70)
        
        for idx, setup in tqdm(signals.iterrows(), total=len(signals), desc="⚡ Processing"):
            pnl, trade_record = self.execute_trade(setup, df, equity)
            equity += pnl
            equity_curve.append(equity)
        
        # Generate reports
        performance = self.generate_performance_report(equity_curve, signals)
        mc_results = self.monte_carlo_simulation()
        mae_mfe = self.analyze_mae_mfe()
        
        # Output results
        self.print_results(performance, mc_results, mae_mfe)
        
        return {
            'performance': performance,
            'monte_carlo': mc_results,
            'mae_mfe': mae_mfe,
            'equity_curve': equity_curve,
            'trade_history': self.trade_history,
            'signals': signals,
            'filtered_signals': self.filtered_signals
        }
    
    def print_results(self, performance, mc_results, mae_mfe):
        """Print comprehensive results."""
        logger.info("\n" + "=" * 70)
        logger.info("📊 BACKTEST RESULTS - TREND-FILTERED STRATEGY")
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
        
        logger.info(f"\n🌡️  TREND ANALYSIS")
        for trend, stats in performance['trend_breakdown'].items():
            logger.info(f"   {trend.replace('_', ' ').title()}:")
            logger.info(f"      Trades: {stats['count']} | Win Rate: {stats['win_rate']*100:.1f}% | PnL: ${stats['total_pnl']:,.2f}")
        
        if mc_results:
            logger.info(f"\n🎲 MONTE CARLO SIMULATION ({MONTE_CARLO_RUNS:,} iterations)")
            logger.info(f"   Expected Final Equity:  ${mc_results['final_equity']['median']:,.2f}")
            logger.info(f"   5th Percentile:         ${mc_results['final_equity']['percentile_5']:,.2f}")
            logger.info(f"   95th Percentile:        ${mc_results['final_equity']['percentile_95']:,.2f}")
            logger.info(f"   Expected Max Drawdown:  {mc_results['max_drawdown']['median']*100:.2f}%")
        
        if mae_mfe:
            logger.info(f"\n📉 MAE/MFE ANALYSIS")
            logger.info(f"   Winners - Avg MAE: ${mae_mfe['winners']['avg_mae']:.2f} | Avg MFE: ${mae_mfe['winners']['avg_mfe']:.2f}")
            logger.info(f"   Losers  - Avg MAE: ${mae_mfe['losers']['avg_mae']:.2f} | Avg MFE: ${mae_mfe['losers']['avg_mfe']:.2f}")
        
        logger.info("\n" + "=" * 70)
        logger.info("✅ BACKTEST COMPLETE")
        logger.info("=" * 70)


if __name__ == "__main__":
    backtest = MokshaOptimizedBacktestV2()
    results = backtest.run()
    
    if results:
        output_file = f"/Users/viveksingh/Documents/Patience/Analysis/TARS/MOKSHA/Moksha_1/moksha_v2_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        serializable_results = {
            'performance': results['performance'],
            'monte_carlo': results['monte_carlo'],
            'mae_mfe': results['mae_mfe'],
            'equity_curve': results['equity_curve'],
            'total_trades': len(results['trade_history']),
            'total_filtered': len(results['filtered_signals']),
            'config': {
                'symbol': SYMBOL,
                'capital': CAPITAL,
                'trend_filter': REQUIRE_TREND_ALIGNMENT,
                'pattern_filter': PROFITABLE_PATTERNS if USE_ONLY_PROFITABLE_PATTERNS else 'ALL',
                'profit_target': f'{TAKE_PROFIT_RATIO}R',
                'volume_filter': f'{MIN_VOLUME_MULTIPLIER}x',
                'time_window': f'10:00-{Q2_END_TIME}'
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"\n💾 Results saved to: {output_file}")