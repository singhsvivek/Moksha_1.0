"""
MOKSHA CAPITAL MANAGEMENT - STRATEGY V3.0: INTRADAY STRUCTURE FOCUS
Learning from V2 Failure: Use Intraday Context for Intraday Patterns

V2 PROBLEM:
- Daily 20-day MA trend filter was wrong timescale for 2-4 hour patterns
- Filtered 94 out of 96 signals (too restrictive)
- 2 trades that passed both lost money (wrong context)

V3 SOLUTION:
- Use INTRADAY market structure instead of daily trends
- Measure Q1 momentum, range position, volume pattern
- Filter patterns based on IMMEDIATE market context
- Allow more trades (target 30-50) with smarter quality control

CORE PHILOSOPHY:
"Intraday patterns need intraday context, not daily context"

Author: Moksha Capital Quantitative Research
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
# CONFIGURATION
# ============================================================================

SYMBOL = 'AGQ'
START_DATE = '2024-01-01'
CAPITAL = 25000.0
BASE_RISK_PCT = 0.02
MAX_RISK_PCT = 0.03
MIN_POSITION_SIZE = 10

# Pattern Selection: Start with all patterns, filter by structure
USE_ONLY_PROFITABLE_PATTERNS = False  # Changed: Test all patterns with structure filter
PROFITABLE_PATTERNS = ['Wick Sweep Long', 'Fakeout Reclaim Short']

# ATR and Risk
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 1.5
TAKE_PROFIT_RATIO = 1.5  # Keep the 1.5R target
BREAKEVEN_RATIO = 1.0

# Transaction Costs
SLIPPAGE_BPS = 1.5
SPREAD_DOLLARS = 0.02
COMMISSION = 0.0

# NEW: Intraday Market Structure Filters (replacing daily trend)
USE_INTRADAY_STRUCTURE_FILTER = True
Q1_RANGE_POSITION_THRESHOLD = 0.35  # Q1 close must be >65% (bullish) or <35% (bearish)
Q1_MOMENTUM_THRESHOLD = 0.003  # 0.3% move to confirm direction
VOLUME_CONFIRMATION_REQUIRED = False  # Start without, can enable later

# Signal Quality (RELAXED from V2)
MIN_VOLUME_MULTIPLIER = 1.2  # Back to 1.2x from 1.5x
MIN_Q1_BARS = 15
MAX_DAILY_TRADES = 3  # Allow 3 trades per day

# Time Filters (EXPANDED from V2)
Q2_START_TIME = "10:00"
Q2_END_TIME = "11:30"  # Back to 11:30, full Q2 window
EARLY_Q2_CUTOFF = "10:30"

# Risk Management
MAX_CONSECUTIVE_LOSSES = 2
POSITION_SCALE_DOWN = 0.50
MAX_DAILY_LOSS_PCT = 0.04
KELLY_FRACTION = 0.25

# Statistical
MONTE_CARLO_RUNS = 5000
MIN_TRADES_FOR_KELLY = 20

# ============================================================================
# MOKSHA V3 BACKTEST ENGINE
# ============================================================================

class MokshaV3IntraydayStructure:
    """
    Version 3: Intraday structure-aware strategy.
    
    Key Innovation: Match pattern timescale with context timescale.
    Use Q1 market structure to filter Q2 patterns.
    """
    
    def __init__(self):
        """Initialize backtest engine."""
        try:
            base_url = getattr(settings, 'ALPACA_BASE_URL', "https://paper-api.alpaca.markets")
            self.api = tradeapi.REST(
                settings.ALPACA_API_KEY,
                settings.ALPACA_SECRET_KEY,
                base_url=base_url,
                api_version='v2'
            )
            logger.info("✅ Alpaca API Connected")
        except Exception as e:
            logger.error(f"❌ Connection Failed: {e}")
            sys.exit(1)
        
        # Trade tracking
        self.trade_history = []
        self.consecutive_losses = 0
        self.daily_pnl = {}
        self.current_position_scale = 1.0
        
        # Analysis storage
        self.mae_mfe_data = []
        self.structure_analysis = []  # NEW: Track Q1 structure decisions
        self.filtered_signals = []
        
    def calculate_transaction_costs(self, shares, entry_price):
        """Calculate realistic transaction costs."""
        position_value = shares * entry_price
        slippage_cost = (position_value * SLIPPAGE_BPS / 10000) * 2
        spread_cost = shares * SPREAD_DOLLARS * 2
        commission_cost = COMMISSION * 2
        return slippage_cost + spread_cost + commission_cost
    
    def calculate_atr(self, df, period=ATR_PERIOD):
        """Calculate Average True Range."""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr
    
    def analyze_q1_structure(self, q1_data):
        """
        Analyze Quarter 1 market structure to determine intraday bias.
        
        This is the KEY innovation in V3. Instead of looking at 20-day trends,
        we look at the last 90 minutes of market action.
        
        Returns:
            structure_bias: 'bullish', 'bearish', or 'neutral'
            confidence: float 0-1 indicating how clear the structure is
            metrics: dict with all calculated metrics for analysis
        """
        if len(q1_data) < 10:
            return 'neutral', 0.5, {}
        
        # Get key prices
        q1_open = q1_data.iloc[0]['open']
        q1_close = q1_data.iloc[-1]['close']
        q1_high = q1_data['high'].max()
        q1_low = q1_data['low'].min()
        q1_range = q1_high - q1_low
        
        if q1_range == 0:  # Avoid division by zero
            return 'neutral', 0.3, {}
        
        # Metric 1: Where did Q1 close within its range?
        # 0 = at low, 1 = at high
        range_position = (q1_close - q1_low) / q1_range
        
        # Metric 2: Net price momentum during Q1
        # Positive = buyers pushed price up, Negative = sellers pushed down
        q1_momentum = (q1_close - q1_open) / q1_open
        
        # Metric 3: Average True Range for context
        q1_atr = q1_data['atr'].iloc[-1] if not pd.isna(q1_data['atr'].iloc[-1]) else q1_range * 0.5
        
        # Metric 4: Volume trend (if available)
        q1_volume = q1_data['volume'].sum()
        q1_avg_volume_per_bar = q1_volume / len(q1_data)
        
        # Determine structure bias
        confidence = 0.0
        structure_bias = 'neutral'
        
        # Strong bullish structure:
        # - Closed in upper 65% of range (range_position > 0.65)
        # - Positive momentum > 0.3%
        if range_position > (1 - Q1_RANGE_POSITION_THRESHOLD) and q1_momentum > Q1_MOMENTUM_THRESHOLD:
            structure_bias = 'bullish'
            # Confidence based on how strong the signals are
            confidence = min(1.0, (range_position - 0.5) * 2 + abs(q1_momentum) / 0.01)
        
        # Strong bearish structure:
        # - Closed in lower 35% of range (range_position < 0.35)
        # - Negative momentum < -0.3%
        elif range_position < Q1_RANGE_POSITION_THRESHOLD and q1_momentum < -Q1_MOMENTUM_THRESHOLD:
            structure_bias = 'bearish'
            confidence = min(1.0, (0.5 - range_position) * 2 + abs(q1_momentum) / 0.01)
        
        # Neutral/choppy structure:
        # - Closed mid-range or momentum doesn't confirm range position
        else:
            structure_bias = 'neutral'
            # Lower confidence in neutral conditions
            confidence = 0.5
        
        metrics = {
            'range_position': range_position,
            'momentum_pct': q1_momentum * 100,
            'q1_range': q1_range,
            'q1_atr': q1_atr,
            'q1_open': q1_open,
            'q1_close': q1_close,
            'q1_high': q1_high,
            'q1_low': q1_low,
            'avg_volume': q1_avg_volume_per_bar
        }
        
        return structure_bias, confidence, metrics
    
    def check_structure_alignment(self, signal_direction, structure_bias, confidence):
        """
        Check if pattern direction aligns with Q1 structure.
        
        LOGIC:
        - Bullish structure → favor LONG patterns, skeptical of SHORT
        - Bearish structure → favor SHORT patterns, skeptical of LONG
        - Neutral structure → allow both but with caution
        
        This is more flexible than V2's hard trend filter.
        """
        if not USE_INTRADAY_STRUCTURE_FILTER:
            return True, 1.0, "structure_filter_disabled"
        
        if signal_direction == 1:  # Long pattern
            if structure_bias == 'bullish':
                # Strongly favor: long pattern with bullish structure
                multiplier = 1.0 + (confidence * 0.3)  # Up to 1.3x sizing
                return True, multiplier, f"aligned_long_in_bullish_{confidence:.2f}"
            
            elif structure_bias == 'neutral':
                # Cautiously allow: structure not clear
                multiplier = 1.0
                return True, multiplier, "neutral_structure_long"
            
            else:  # bearish structure
                # Skeptical but don't reject: might be contrarian opportunity
                # Reduce size but allow trade
                multiplier = 0.7
                return True, multiplier, "contrarian_long_in_bearish"
        
        elif signal_direction == -1:  # Short pattern
            if structure_bias == 'bearish':
                # Strongly favor: short pattern with bearish structure
                multiplier = 1.0 + (confidence * 0.3)
                return True, multiplier, f"aligned_short_in_bearish_{confidence:.2f}"
            
            elif structure_bias == 'neutral':
                # Cautiously allow
                multiplier = 1.0
                return True, multiplier, "neutral_structure_short"
            
            else:  # bullish structure
                # Skeptical but allow with reduced size
                multiplier = 0.7
                return True, multiplier, "contrarian_short_in_bullish"
        
        return False, 1.0, "invalid_signal"
    
    def fetch_real_data(self):
        """Fetch historical data."""
        logger.info(f"⏳ Fetching Data for {SYMBOL}...")
        end_dt = datetime.now()
        start_dt = end_dt - timedelta(days=365 + 30)  # Extra days for ATR warmup
        
        try:
            bars = self.api.get_bars(
                SYMBOL,
                '5Min',
                start=start_dt.strftime('%Y-%m-%d'),
                end=end_dt.strftime('%Y-%m-%d'),
                feed='iex'
            ).df
            
            if bars.empty:
                raise ValueError("No data returned")
            
            if bars.index.tz is None:
                bars.index = bars.index.tz_localize('UTC')
            bars.index = bars.index.tz_convert('America/Chicago')
            
            bars['atr'] = self.calculate_atr(bars)
            
            logger.info(f"✅ Fetched {len(bars)} bars from {bars.index[0]} to {bars.index[-1]}")
            
            return bars
            
        except Exception as e:
            logger.exception(f"❌ Data Fetch Error: {e}")
            return pd.DataFrame()
    
    def identify_trap_patterns(self, df):
        """
        Pattern identification with intraday structure filtering.
        
        MAJOR CHANGE FROM V2:
        - Analyze Q1 structure for each day
        - Use Q1 structure to inform Q2 pattern trading
        - Don't reject patterns, just adjust sizing based on alignment
        """
        df['time_only'] = df.index.time
        t_open = pd.Timestamp("08:30").time()
        t_q1_end = pd.Timestamp("10:00").time()
        t_q2_end = pd.Timestamp(Q2_END_TIME).time()
        t_early_q2 = pd.Timestamp(EARLY_Q2_CUTOFF).time()
        
        signals = []
        grouped = df.groupby(df.index.date)
        
        logger.info(f"\n{'='*70}")
        logger.info(f"🔍 V3: INTRADAY STRUCTURE-BASED PATTERN SCANNING")
        logger.info(f"{'='*70}")
        logger.info(f"Structure Filter: {'ENABLED' if USE_INTRADAY_STRUCTURE_FILTER else 'DISABLED'}")
        logger.info(f"Pattern Mode: {'SELECTIVE' if USE_ONLY_PROFITABLE_PATTERNS else 'ALL PATTERNS'}")
        logger.info(f"Q1 Range Threshold: {Q1_RANGE_POSITION_THRESHOLD} ({(1-Q1_RANGE_POSITION_THRESHOLD)*100:.0f}% for bullish)")
        logger.info(f"Q1 Momentum Threshold: {Q1_MOMENTUM_THRESHOLD*100:.1f}%")
        logger.info(f"Volume Filter: {MIN_VOLUME_MULTIPLIER}x")
        logger.info(f"{'='*70}\n")
        
        for date, day_data in tqdm(grouped, desc="📊 Analyzing Days"):
            if len(day_data) < 30:
                continue
            
            if self.check_daily_loss_limit(pd.Timestamp(date)):
                continue
            
            daily_trades = len([s for s in signals if s['time'].date() == date])
            if daily_trades >= MAX_DAILY_TRADES:
                continue
            
            # ==================================================================
            # Q1 ANALYSIS - This is where V3 diverges from V2
            # ==================================================================
            q1_data = day_data[
                (day_data['time_only'] >= t_open) & 
                (day_data['time_only'] < t_q1_end)
            ]
            
            if len(q1_data) < MIN_Q1_BARS:
                continue
            
            # Analyze Q1 market structure
            structure_bias, structure_confidence, structure_metrics = self.analyze_q1_structure(q1_data)
            
            # Store for analysis
            self.structure_analysis.append({
                'date': date,
                'bias': structure_bias,
                'confidence': structure_confidence,
                **structure_metrics
            })
            
            q1_high = structure_metrics['q1_high']
            q1_low = structure_metrics['q1_low']
            q1_avg_volume = structure_metrics['avg_volume']
            
            # ==================================================================
            # Q2 PATTERN SCANNING
            # ==================================================================
            q2_data = day_data[
                (day_data['time_only'] >= t_q1_end) & 
                (day_data['time_only'] < t_q2_end)
            ]
            
            if q2_data.empty:
                continue
            
            current_atr = q2_data['atr'].iloc[0]
            if pd.isna(current_atr):
                current_atr = structure_metrics['q1_range'] * 0.5
            
            candles = q2_data.rename_axis('timestamp').reset_index().to_dict('records')
            
            for i in range(len(candles)):
                curr = candles[i]
                prev = candles[i-1] if i > 0 else None
                
                signal = 0
                setup_type = ""
                stop_loss = 0
                confidence = 1.0
                filter_reason = None
                
                # Pattern Detection (same as before)
                if curr['low'] < q1_low and curr['close'] > q1_low:
                    signal = 1
                    setup_type = "Wick Sweep Long"
                    stop_loss = curr['low'] - (current_atr * ATR_STOP_MULTIPLIER)
                    
                    if curr['volume'] < q1_avg_volume * MIN_VOLUME_MULTIPLIER:
                        confidence *= 0.8
                
                elif curr['high'] > q1_high and curr['close'] < q1_high:
                    signal = -1
                    setup_type = "Wick Sweep Short"
                    stop_loss = curr['high'] + (current_atr * ATR_STOP_MULTIPLIER)
                    
                    if curr['volume'] < q1_avg_volume * MIN_VOLUME_MULTIPLIER:
                        confidence *= 0.8
                
                if signal == 0 and prev:
                    if prev['close'] < q1_low and curr['close'] > q1_low:
                        signal = 1
                        setup_type = "Fakeout Reclaim Long"
                        stop_loss = min(prev['low'], curr['low']) - (current_atr * ATR_STOP_MULTIPLIER)
                        
                        if curr['volume'] < q1_avg_volume * MIN_VOLUME_MULTIPLIER:
                            confidence *= 0.8
                    
                    elif prev['close'] > q1_high and curr['close'] < q1_high:
                        signal = -1
                        setup_type = "Fakeout Reclaim Short"
                        stop_loss = max(prev['high'], curr['high']) + (current_atr * ATR_STOP_MULTIPLIER)
                        
                        if curr['volume'] < q1_avg_volume * MIN_VOLUME_MULTIPLIER:
                            confidence *= 0.8
                
                if signal == 0:
                    continue
                
                # ==================================================================
                # FILTERS - More lenient than V2
                # ==================================================================
                
                # Pattern selectivity (if enabled)
                if USE_ONLY_PROFITABLE_PATTERNS:
                    if setup_type not in PROFITABLE_PATTERNS:
                        filter_reason = f"pattern_not_in_list"
                        self.filtered_signals.append({
                            'date': date,
                            'time': curr['timestamp'],
                            'type': setup_type,
                            'reason': filter_reason,
                            'structure_bias': structure_bias
                        })
                        continue
                
                # NEW: Structure alignment check (doesn't reject, just adjusts sizing)
                structure_aligned, structure_multiplier, structure_status = self.check_structure_alignment(
                    signal, structure_bias, structure_confidence
                )
                
                # Don't reject based on structure, just note it
                # This is KEY difference from V2
                
                # Time-of-day confidence
                entry_time = curr['timestamp'].time()
                if entry_time < t_early_q2:
                    confidence *= 1.1
                elif entry_time > pd.Timestamp("11:00").time():
                    confidence *= 0.95
                
                # Store signal
                signals.append({
                    'time': curr['timestamp'],
                    'signal': signal,
                    'entry': curr['close'],
                    'stop': stop_loss,
                    'type': setup_type,
                    'atr': current_atr,
                    'confidence': confidence,
                    'structure_bias': structure_bias,
                    'structure_confidence': structure_confidence,
                    'structure_multiplier': structure_multiplier,
                    'structure_status': structure_status,
                    'volume': curr['volume'],
                    'volume_ratio': curr['volume'] / q1_avg_volume,
                    'q1_high': q1_high,
                    'q1_low': q1_low,
                    'q1_range_position': structure_metrics['range_position'],
                    'q1_momentum': structure_metrics['momentum_pct']
                })
                
                if "Wick Sweep" in setup_type:
                    break
        
        logger.info(f"\n✅ Pattern Scan Complete:")
        logger.info(f"   Signals Found: {len(signals)}")
        logger.info(f"   Signals Filtered: {len(self.filtered_signals)}")
        
        # Structure breakdown
        if self.structure_analysis:
            structure_counts = {}
            for sa in self.structure_analysis:
                bias = sa['bias']
                structure_counts[bias] = structure_counts.get(bias, 0) + 1
            
            logger.info(f"\n   Q1 Structure Distribution:")
            for bias, count in structure_counts.items():
                pct = (count / len(self.structure_analysis)) * 100
                logger.info(f"      {bias.title()}: {count} days ({pct:.1f}%)")
        
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
        """Check daily loss limit."""
        date_key = current_date.date()
        daily_pnl = self.daily_pnl.get(date_key, 0.0)
        return daily_pnl < -CAPITAL * MAX_DAILY_LOSS_PCT
    
    def calculate_kelly_position_size(self):
        """Kelly Criterion position sizing."""
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
        """Execute trade with structure-aware position sizing."""
        entry = setup['entry']
        stop = setup['stop']
        direction = setup['signal']
        setup_type = setup['type']
        confidence = setup['confidence']
        structure_multiplier = setup['structure_multiplier']
        
        # Position sizing with structure awareness
        kelly_risk = self.calculate_kelly_position_size()
        adjusted_risk = kelly_risk * confidence * structure_multiplier
        adjusted_risk *= self.current_position_scale
        
        risk_amount = equity * adjusted_risk
        stop_distance = abs(entry - stop)
        shares = risk_amount / stop_distance
        
        if shares < MIN_POSITION_SIZE:
            shares = MIN_POSITION_SIZE
        else:
            shares = int(shares)
        
        transaction_cost = self.calculate_transaction_costs(shares, entry)
        
        take_profit = entry + (stop_distance * TAKE_PROFIT_RATIO * direction)
        breakeven_trigger = entry + (stop_distance * BREAKEVEN_RATIO * direction)
        
        # Trade execution
        future_data = df[df.index > setup['time']].head(48)
        
        outcome = 0
        stop_moved_to_breakeven = False
        exit_price = entry
        exit_reason = "time"
        
        max_favorable = 0
        max_adverse = 0
        
        for idx, bar in future_data.iterrows():
            current_stop = entry if stop_moved_to_breakeven else stop
            
            if direction == 1:
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
            
            else:
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
        
        if outcome == 0 and not future_data.empty:
            exit_price = future_data.iloc[-1]['close']
            if stop_moved_to_breakeven:
                if (direction == 1 and exit_price < entry) or (direction == -1 and exit_price > entry):
                    exit_price = entry
                    exit_reason = "breakeven"
        
        gross_pnl = (exit_price - entry) * shares * direction
        net_pnl = gross_pnl - transaction_cost
        
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
            'structure_bias': setup['structure_bias'],
            'structure_multiplier': structure_multiplier,
            'confidence': confidence,
            'mae': max_adverse,
            'mfe': max_favorable,
            'risk_pct': adjusted_risk,
            'q1_range_position': setup['q1_range_position'],
            'q1_momentum': setup['q1_momentum']
        }
        
        self.trade_history.append(trade_record)
        
        trade_date = setup['time'].date()
        self.daily_pnl[trade_date] = self.daily_pnl.get(trade_date, 0.0) + net_pnl
        
        # Risk scaling
        if net_pnl < 0:
            self.consecutive_losses += 1
            if self.consecutive_losses >= MAX_CONSECUTIVE_LOSSES:
                self.current_position_scale = POSITION_SCALE_DOWN
        else:
            if self.consecutive_losses >= MAX_CONSECUTIVE_LOSSES:
                self.current_position_scale = 1.0
            self.consecutive_losses = 0
        
        self.mae_mfe_data.append({
            'mae': max_adverse,
            'mfe': max_favorable,
            'outcome': outcome,
            'type': setup_type,
            'structure_bias': setup['structure_bias']
        })
        
        return net_pnl, trade_record
    
    def monte_carlo_simulation(self):
        """Monte Carlo simulation."""
        if len(self.trade_history) < 30:
            return None
        
        logger.info(f"🎲 Running Monte Carlo ({MONTE_CARLO_RUNS:,} iterations)...")
        
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
        
        return {
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
    
    def analyze_mae_mfe(self):
        """MAE/MFE analysis."""
        if not self.mae_mfe_data:
            return None
        
        df = pd.DataFrame(self.mae_mfe_data)
        
        winners = df[df['outcome'] == 1]
        losers = df[df['outcome'] == -1]
        
        return {
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
    
    def generate_performance_report(self, equity_curve, signals_df):
        """Generate performance report."""
        if not self.trade_history:
            return None
        
        total_return = (equity_curve[-1] - CAPITAL) / CAPITAL
        
        winners = [t for t in self.trade_history if t['pnl'] > 0]
        losers = [t for t in self.trade_history if t['pnl'] < 0]
        win_rate = len(winners) / len(self.trade_history)
        
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
        
        # Structure breakdown
        structure_stats = {}
        for structure in ['bullish', 'bearish', 'neutral']:
            trades = [t for t in self.trade_history if t['structure_bias'] == structure]
            if trades:
                wins = [t for t in trades if t['pnl'] > 0]
                structure_stats[structure] = {
                    'count': len(trades),
                    'win_rate': len(wins) / len(trades),
                    'total_pnl': sum([t['pnl'] for t in trades])
                }
        
        return {
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
            'structure_breakdown': structure_stats
        }
    
    def run(self):
        """Main execution."""
        logger.info("=" * 70)
        logger.info("🚀 MOKSHA V3 - INTRADAY STRUCTURE FOCUS")
        logger.info("=" * 70)
        
        df = self.fetch_real_data()
        if df.empty:
            return None
        
        signals = self.identify_trap_patterns(df)
        
        if signals.empty:
            logger.error("❌ No signals found")
            return None
        
        logger.info(f"✅ Found {len(signals)} signals")
        
        # Execute trades
        equity = CAPITAL
        equity_curve = [CAPITAL]
        
        logger.info("\n" + "=" * 70)
        logger.info("💼 EXECUTING TRADES")
        logger.info("=" * 70)
        
        for idx, setup in tqdm(signals.iterrows(), total=len(signals), desc="⚡ Trading"):
            pnl, trade_record = self.execute_trade(setup, df, equity)
            equity += pnl
            equity_curve.append(equity)
        
        # Analysis
        performance = self.generate_performance_report(equity_curve, signals)
        mc_results = self.monte_carlo_simulation()
        mae_mfe = self.analyze_mae_mfe()
        
        # Print results
        self.print_results(performance, mc_results, mae_mfe)
        
        return {
            'performance': performance,
            'monte_carlo': mc_results,
            'mae_mfe': mae_mfe,
            'equity_curve': equity_curve,
            'trade_history': self.trade_history,
            'signals': signals,
            'filtered_signals': self.filtered_signals,
            'structure_analysis': self.structure_analysis
        }
    
    def print_results(self, performance, mc_results, mae_mfe):
        """Print results."""
        logger.info("\n" + "=" * 70)
        logger.info("📊 V3 RESULTS - INTRADAY STRUCTURE STRATEGY")
        logger.info("=" * 70)
        
        logger.info(f"\n💰 CAPITAL")
        logger.info(f"   Starting:     ${performance['capital']['starting']:,.2f}")
        logger.info(f"   Ending:       ${performance['capital']['ending']:,.2f}")
        logger.info(f"   Return:       {performance['capital']['total_return']*100:.2f}%")
        logger.info(f"   Profit:       ${performance['capital']['total_pnl']:,.2f}")
        
        logger.info(f"\n📈 TRADES")
        logger.info(f"   Total:    {performance['trades']['total']}")
        logger.info(f"   Winners:  {performance['trades']['winners']}")
        logger.info(f"   Losers:   {performance['trades']['losers']}")
        logger.info(f"   Win Rate: {performance['performance']['win_rate']*100:.1f}%")
        logger.info(f"   Profit Factor: {performance['performance']['profit_factor']:.2f}")
        logger.info(f"   Expectancy: ${performance['performance']['expectancy']:.2f}")
        
        logger.info(f"\n💵 WIN/LOSS")
        logger.info(f"   Avg Win:  ${performance['performance']['avg_win']:.2f}")
        logger.info(f"   Avg Loss: ${performance['performance']['avg_loss']:.2f}")
        logger.info(f"   Avg Cost: ${performance['risk']['avg_transaction_cost']:.2f}")
        
        logger.info(f"\n⚠️  RISK")
        logger.info(f"   Max DD:     {performance['risk']['max_drawdown']*100:.2f}%")
        logger.info(f"   Sharpe:     {performance['risk']['sharpe_ratio']:.2f}")
        
        logger.info(f"\n🎯 PATTERN BREAKDOWN")
        for strategy, stats in performance['strategy_breakdown'].items():
            logger.info(f"   {strategy}:")
            logger.info(f"      {stats['count']} trades | {stats['win_rate']*100:.1f}% WR | ${stats['total_pnl']:,.2f}")
        
        logger.info(f"\n🏗️  STRUCTURE BREAKDOWN")
        for structure, stats in performance['structure_breakdown'].items():
            logger.info(f"   {structure.title()}:")
            logger.info(f"      {stats['count']} trades | {stats['win_rate']*100:.1f}% WR | ${stats['total_pnl']:,.2f}")
        
        if mc_results:
            logger.info(f"\n🎲 MONTE CARLO ({MONTE_CARLO_RUNS:,} runs)")
            logger.info(f"   Median:  ${mc_results['final_equity']['median']:,.2f}")
            logger.info(f"   5th %:   ${mc_results['final_equity']['percentile_5']:,.2f}")
            logger.info(f"   95th %:  ${mc_results['final_equity']['percentile_95']:,.2f}")
        
        if mae_mfe:
            logger.info(f"\n📉 MAE/MFE")
            logger.info(f"   Winners: MAE ${mae_mfe['winners']['avg_mae']:.2f} | MFE ${mae_mfe['winners']['avg_mfe']:.2f}")
            logger.info(f"   Losers:  MAE ${mae_mfe['losers']['avg_mae']:.2f} | MFE ${mae_mfe['losers']['avg_mfe']:.2f}")
        
        logger.info("\n" + "=" * 70)
        logger.info("✅ BACKTEST COMPLETE")
        logger.info("=" * 70)


if __name__ == "__main__":
    backtest = MokshaV3IntraydayStructure()
    results = backtest.run()
    
    if results:
        
        output_file = f"/Users/viveksingh/Documents/Patience/Analysis/TARS/MOKSHA/Moksha_1/moksha_v3_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        output_file = f"/Users/viveksingh/Documents/Patience/Analysis/TARS/MOKSHA/Moksha_1/moksha_v2_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        serializable_results = {
            'performance': results['performance'],
            'monte_carlo': results['monte_carlo'],
            'mae_mfe': results['mae_mfe'],
            'equity_curve': results['equity_curve'],
            'total_trades': len(results['trade_history']),
            'total_filtered': len(results['filtered_signals']),
            'structure_days': len(results['structure_analysis']),
            'config': {
                'symbol': SYMBOL,
                'capital': CAPITAL,
                'structure_filter': USE_INTRADAY_STRUCTURE_FILTER,
                'pattern_filter': PROFITABLE_PATTERNS if USE_ONLY_PROFITABLE_PATTERNS else 'ALL',
                'q1_range_threshold': Q1_RANGE_POSITION_THRESHOLD,
                'q1_momentum_threshold': Q1_MOMENTUM_THRESHOLD,
                'volume_filter': f'{MIN_VOLUME_MULTIPLIER}x'
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"\n💾 Results saved: {output_file}")