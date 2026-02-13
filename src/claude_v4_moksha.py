"""
MOKSHA CAPITAL MANAGEMENT - STRATEGY V4.0: NSWL FOCUSED
Neutral Structure Wick Sweep Long - The Evidence-Based Approach

LEARNING JOURNEY TO V4:
------------------------
V1 Result: -$79 (-0.32%) | 101 trades | 37.6% WR
   Learning: Wick Sweep Long worked (+$2,688), shorts failed catastrophically

V2 Result: -$2,319 (-9.3%) | 2 trades | 0% WR  
   Learning: Daily trend filter wrong timescale, over-filtered to death

V3 Result: -$1,481 (-5.9%) | 88 trades | 40.9% WR
   Learning: CRITICAL DISCOVERY - Neutral structure performed BEST
   
   Structure Breakdown V3:
   - Bullish: 41 trades, 36.6% WR, -$2,243 (WORST)
   - Bearish: 23 trades, 39.1% WR, -$454 (BAD)
   - Neutral: 24 trades, 50.0% WR, +$1,215 (PROFITABLE!)
   
   This completely inverted our hypothesis. Mean reversion patterns work best
   when Q1 shows NO clear direction, not when it shows strong momentum.

V4 STRATEGY:
------------
Trade ONLY when Q1 is choppy/neutral (no strong directional bias)
Trade ONLY Wick Sweep Long pattern (the only consistently profitable one)
Trade ONLY in first 30 minutes of Q2 (10:00-10:30)
Use tighter profit targets (1.25R instead of 1.5R)
Add range contraction filter (Q1 range < ATR means compression)

HYPOTHESIS:
Trap patterns exploit market noise, not directional moves. They work best
when the market is confused and indecisive, not when it has momentum.

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
# CONFIGURATION - FOCUSED AND EVIDENCE-BASED
# ============================================================================

SYMBOL = 'AGQ'
START_DATE = '2024-01-01'
CAPITAL = 25000.0
BASE_RISK_PCT = 0.015  # Reduced from 2% to 1.5% for focused single-pattern approach
MAX_RISK_PCT = 0.025
MIN_POSITION_SIZE = 10

# CRITICAL: Pattern Selection - ONLY what works
TRADE_ONLY_WICK_SWEEP_LONG = True  # The ONLY pattern that consistently works

# ATR and Risk Management
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 1.5
TAKE_PROFIT_RATIO = 1.25  # REDUCED from 1.5R - accept smaller, more frequent wins
BREAKEVEN_RATIO = 0.75     # REDUCED - move to BE faster (was 1.0)

# Transaction Costs (based on actual results averaging $23)
SLIPPAGE_BPS = 1.5
SPREAD_DOLLARS = 0.02
COMMISSION = 0.0

# NEW V4: NEUTRAL STRUCTURE FILTERS (inverted from V3!)
USE_NEUTRAL_STRUCTURE_FILTER = True
NEUTRAL_RANGE_MIN = 0.35  # Q1 close must be between 35% and 65% of range
NEUTRAL_RANGE_MAX = 0.65  # (middle 30% indicates choppy, no clear winner)
NEUTRAL_MOMENTUM_MAX = 0.002  # Momentum must be < 0.2% (was 0.3% for directional)

# NEW: Range Contraction Filter
REQUIRE_RANGE_CONTRACTION = True
MAX_Q1_RANGE_VS_ATR = 1.0  # Q1 range must be < 100% of ATR (compressed, not expanded)

# NEW: Strict Early Q2 Window
TRADE_ONLY_EARLY_Q2 = True
Q2_START_TIME = "10:00"
Q2_END_TIME = "10:30"  # STRICT 30-minute window only

# Signal Quality (balanced)
MIN_VOLUME_MULTIPLIER = 1.2
MIN_Q1_BARS = 15
MAX_DAILY_TRADES = 2  # Very selective, max 2 per day

# Risk Management
MAX_CONSECUTIVE_LOSSES = 2
POSITION_SCALE_DOWN = 0.50
MAX_DAILY_LOSS_PCT = 0.04
KELLY_FRACTION = 0.25

# Statistical
MONTE_CARLO_RUNS = 5000
MIN_TRADES_FOR_KELLY = 15  # Reduced since we'll have fewer trades

# ============================================================================
# MOKSHA V4: NEUTRAL STRUCTURE FOCUSED ENGINE
# ============================================================================

class MokshaV4NeutralStructure:
    """
    Version 4: Evidence-based focused strategy.
    
    Core Insight from V3:
    Mean reversion patterns work in NOISE, not MOMENTUM.
    Trade only when Q1 is choppy and only the proven pattern.
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
        
        self.trade_history = []
        self.consecutive_losses = 0
        self.daily_pnl = {}
        self.current_position_scale = 1.0
        
        self.mae_mfe_data = []
        self.structure_analysis = []
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
        Analyze Q1 to determine if conditions are NEUTRAL/CHOPPY.
        
        V4 CRITICAL CHANGE:
        We now WANT neutral/choppy conditions, not directional ones.
        V3 showed neutral structure was the ONLY profitable category.
        
        Returns:
            is_neutral: True if Q1 is choppy/neutral
            metrics: All calculated metrics for analysis
        """
        if len(q1_data) < 10:
            return False, {}
        
        q1_open = q1_data.iloc[0]['open']
        q1_close = q1_data.iloc[-1]['close']
        q1_high = q1_data['high'].max()
        q1_low = q1_data['low'].min()
        q1_range = q1_high - q1_low
        
        if q1_range == 0:
            return False, {}
        
        # Where in range did Q1 close? (0 = low, 1 = high)
        range_position = (q1_close - q1_low) / q1_range
        
        # How much momentum during Q1?
        q1_momentum = (q1_close - q1_open) / q1_open
        
        # ATR for context
        q1_atr = q1_data['atr'].iloc[-1] if not pd.isna(q1_data['atr'].iloc[-1]) else q1_range * 0.5
        
        # Range size relative to ATR
        range_vs_atr = q1_range / q1_atr if q1_atr > 0 else 1.0
        
        # Volume
        q1_volume = q1_data['volume'].sum()
        q1_avg_volume = q1_volume / len(q1_data)
        
        # ====================================================================
        # V4 NEUTRAL STRUCTURE CRITERIA
        # ====================================================================
        
        # Condition 1: Range position must be in middle (neutral close)
        in_neutral_range = (range_position >= NEUTRAL_RANGE_MIN and 
                          range_position <= NEUTRAL_RANGE_MAX)
        
        # Condition 2: Momentum must be small (no strong direction)
        low_momentum = abs(q1_momentum) < NEUTRAL_MOMENTUM_MAX
        
        # Condition 3: Range should not be expanded (optional but recommended)
        if REQUIRE_RANGE_CONTRACTION:
            range_ok = range_vs_atr < MAX_Q1_RANGE_VS_ATR
        else:
            range_ok = True
        
        # Q1 is neutral if ALL conditions met
        is_neutral = in_neutral_range and low_momentum and range_ok
        
        # Classify for tracking (opposite of V3 logic)
        if in_neutral_range and low_momentum:
            structure_type = 'neutral'  # This is what we WANT now
        elif range_position > 0.65 and q1_momentum > 0.003:
            structure_type = 'bullish'  # This we now AVOID
        elif range_position < 0.35 and q1_momentum < -0.003:
            structure_type = 'bearish'  # This we now AVOID
        else:
            structure_type = 'unclear'
        
        metrics = {
            'range_position': range_position,
            'momentum_pct': q1_momentum * 100,
            'q1_range': q1_range,
            'q1_atr': q1_atr,
            'range_vs_atr': range_vs_atr,
            'q1_open': q1_open,
            'q1_close': q1_close,
            'q1_high': q1_high,
            'q1_low': q1_low,
            'avg_volume': q1_avg_volume,
            'structure_type': structure_type,
            'is_neutral': is_neutral,
            'in_neutral_range': in_neutral_range,
            'low_momentum': low_momentum,
            'range_ok': range_ok
        }
        
        return is_neutral, metrics
    
    def fetch_real_data(self):
        """Fetch historical data."""
        logger.info(f"⏳ Fetching Data for {SYMBOL}...")
        end_dt = datetime.now()
        start_dt = end_dt - timedelta(days=365 + 30)
        
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
            
            logger.info(f"✅ Fetched {len(bars)} bars")
            return bars
            
        except Exception as e:
            logger.exception(f"❌ Data Fetch Error: {e}")
            return pd.DataFrame()
    
    def identify_wick_sweep_long_patterns(self, df):
        """
        Pattern identification: ONLY Wick Sweep Long in neutral Q1 conditions.
        
        This is the most focused version yet. We're looking for one specific
        pattern that has demonstrated edge, occurring only in the specific
        market conditions where V3 showed it actually works.
        """
        df['time_only'] = df.index.time
        t_open = pd.Timestamp("08:30").time()
        t_q1_end = pd.Timestamp("10:00").time()
        t_q2_start = pd.Timestamp(Q2_START_TIME).time()
        t_q2_end = pd.Timestamp(Q2_END_TIME).time()
        
        signals = []
        grouped = df.groupby(df.index.date)
        
        logger.info(f"\n{'='*70}")
        logger.info(f"🎯 V4: NEUTRAL STRUCTURE WICK SWEEP LONG ONLY")
        logger.info(f"{'='*70}")
        logger.info(f"Pattern: Wick Sweep Long ONLY")
        logger.info(f"Structure: Neutral (choppy Q1) ONLY")
        logger.info(f"Time Window: {Q2_START_TIME}-{Q2_END_TIME} (30 min)")
        logger.info(f"Range Filter: Q1 range < {MAX_Q1_RANGE_VS_ATR}x ATR")
        logger.info(f"Target: {TAKE_PROFIT_RATIO}R (tighter than V3)")
        logger.info(f"{'='*70}\n")
        
        neutral_days_count = 0
        directional_days_count = 0
        
        for date, day_data in tqdm(grouped, desc="🔬 Scanning for Optimal Conditions"):
            if len(day_data) < 30:
                continue
            
            if self.check_daily_loss_limit(pd.Timestamp(date)):
                continue
            
            daily_trades = len([s for s in signals if s['time'].date() == date])
            if daily_trades >= MAX_DAILY_TRADES:
                continue
            
            # ================================================================
            # Q1 ANALYSIS - Looking for NEUTRAL structure
            # ================================================================
            q1_data = day_data[
                (day_data['time_only'] >= t_open) & 
                (day_data['time_only'] < t_q1_end)
            ]
            
            if len(q1_data) < MIN_Q1_BARS:
                continue
            
            # Check if Q1 is neutral/choppy
            is_neutral, q1_metrics = self.analyze_q1_structure(q1_data)
            
            # Store for analysis
            self.structure_analysis.append({
                'date': date,
                'is_neutral': is_neutral,
                **q1_metrics
            })
            
            # V4 CRITICAL FILTER: Only trade on neutral days
            if not is_neutral:
                directional_days_count += 1
                # Log why we're skipping this day
                self.filtered_signals.append({
                    'date': date,
                    'reason': f"not_neutral_{q1_metrics['structure_type']}",
                    'range_position': q1_metrics['range_position'],
                    'momentum': q1_metrics['momentum_pct'],
                    'range_vs_atr': q1_metrics['range_vs_atr']
                })
                continue
            
            neutral_days_count += 1
            
            # ================================================================
            # Q2 PATTERN SCANNING - Only Wick Sweep Long
            # ================================================================
            q2_data = day_data[
                (day_data['time_only'] >= t_q2_start) & 
                (day_data['time_only'] < t_q2_end)
            ]
            
            if q2_data.empty:
                continue
            
            current_atr = q2_data['atr'].iloc[0]
            if pd.isna(current_atr):
                current_atr = q1_metrics['q1_range'] * 0.5
            
            q1_high = q1_metrics['q1_high']
            q1_low = q1_metrics['q1_low']
            q1_avg_volume = q1_metrics['avg_volume']
            
            candles = q2_data.rename_axis('timestamp').reset_index().to_dict('records')
            
            for i in range(len(candles)):
                curr = candles[i]
                
                # ============================================================
                # PATTERN: Wick Sweep Long ONLY
                # ============================================================
                # Price sweeps BELOW Q1 low but closes ABOVE it
                # This is a bullish rejection - trap pattern
                
                if curr['low'] < q1_low and curr['close'] > q1_low:
                    signal = 1  # Long
                    setup_type = "Wick Sweep Long"
                    stop_loss = curr['low'] - (current_atr * ATR_STOP_MULTIPLIER)
                    
                    # Volume confirmation
                    if curr['volume'] < q1_avg_volume * MIN_VOLUME_MULTIPLIER:
                        # Don't reject, just note lower confidence
                        confidence = 0.85
                    else:
                        confidence = 1.0
                    
                    # Store signal
                    signals.append({
                        'time': curr['timestamp'],
                        'signal': signal,
                        'entry': curr['close'],
                        'stop': stop_loss,
                        'type': setup_type,
                        'atr': current_atr,
                        'confidence': confidence,
                        'volume': curr['volume'],
                        'volume_ratio': curr['volume'] / q1_avg_volume,
                        'q1_high': q1_high,
                        'q1_low': q1_low,
                        'q1_range_position': q1_metrics['range_position'],
                        'q1_momentum': q1_metrics['momentum_pct'],
                        'q1_range_vs_atr': q1_metrics['range_vs_atr'],
                        'structure_type': q1_metrics['structure_type']
                    })
                    
                    # Only take first pattern each day
                    break
        
        logger.info(f"\n✅ Scan Complete:")
        logger.info(f"   Neutral Days Found: {neutral_days_count}")
        logger.info(f"   Directional Days Skipped: {directional_days_count}")
        logger.info(f"   Signals Found: {len(signals)}")
        logger.info(f"   Signals Filtered: {len(self.filtered_signals)}")
        
        if self.filtered_signals:
            filter_counts = {}
            for fs in self.filtered_signals:
                reason = fs['reason']
                filter_counts[reason] = filter_counts.get(reason, 0) + 1
            
            logger.info(f"\n   Days Skipped By Reason:")
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
        
        recent_trades = self.trade_history[-30:]  # Use last 30 trades
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
        Execute trade with tighter profit targets.
        
        V4 uses 1.25R target instead of 1.5R, and moves to breakeven
        at 0.75R instead of 1R. This captures profits more reliably.
        """
        entry = setup['entry']
        stop = setup['stop']
        direction = setup['signal']  # Always 1 (long) in V4
        setup_type = setup['type']
        confidence = setup['confidence']
        
        # Position sizing
        kelly_risk = self.calculate_kelly_position_size()
        adjusted_risk = kelly_risk * confidence
        adjusted_risk *= self.current_position_scale
        
        risk_amount = equity * adjusted_risk
        stop_distance = abs(entry - stop)
        shares = risk_amount / stop_distance
        
        if shares < MIN_POSITION_SIZE:
            shares = MIN_POSITION_SIZE
        else:
            shares = int(shares)
        
        transaction_cost = self.calculate_transaction_costs(shares, entry)
        
        # V4 TARGETS: Tighter profit taking
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
            
            # Long only in V4
            favorable_move = bar['high'] - entry
            adverse_move = entry - bar['low']
            max_favorable = max(max_favorable, favorable_move)
            max_adverse = max(max_adverse, adverse_move)
            
            # Check stop
            if bar['low'] <= current_stop:
                exit_price = current_stop
                outcome = -1 if not stop_moved_to_breakeven else 0
                exit_reason = "stop_loss" if not stop_moved_to_breakeven else "breakeven"
                break
            
            # Check target (1.25R)
            if bar['high'] >= take_profit:
                exit_price = take_profit
                outcome = 1
                exit_reason = "take_profit"
                break
            
            # Move to breakeven at 0.75R
            if bar['high'] >= breakeven_trigger and not stop_moved_to_breakeven:
                stop_moved_to_breakeven = True
        
        # Time exit
        if outcome == 0 and not future_data.empty:
            exit_price = future_data.iloc[-1]['close']
            if stop_moved_to_breakeven:
                if exit_price < entry:
                    exit_price = entry
                    exit_reason = "breakeven"
        
        # PnL
        gross_pnl = (exit_price - entry) * shares * direction
        net_pnl = gross_pnl - transaction_cost
        
        # Record
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
            'confidence': confidence,
            'mae': max_adverse,
            'mfe': max_favorable,
            'risk_pct': adjusted_risk,
            'q1_range_position': setup['q1_range_position'],
            'q1_momentum': setup['q1_momentum'],
            'q1_range_vs_atr': setup['q1_range_vs_atr']
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
            'type': setup_type
        })
        
        return net_pnl, trade_record
    
    def monte_carlo_simulation(self):
        """Monte Carlo simulation."""
        if len(self.trade_history) < 15:  # Need at least 15 trades
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
            }
        }
    
    def run(self):
        """Main execution."""
        logger.info("=" * 70)
        logger.info("🚀 MOKSHA V4 - NEUTRAL STRUCTURE WICK SWEEP LONG")
        logger.info("=" * 70)
        logger.info("Evidence-Based Focused Strategy")
        logger.info("Trading ONLY what works, ONLY when it works")
        logger.info("=" * 70)
        
        df = self.fetch_real_data()
        if df.empty:
            return None
        
        signals = self.identify_wick_sweep_long_patterns(df)
        
        if signals.empty:
            logger.error("❌ No signals found in neutral conditions")
            logger.info("\n💡 This means Q1 rarely showed neutral structure in your test period.")
            logger.info("   Consider relaxing neutral criteria or testing different time period.")
            return None
        
        logger.info(f"✅ Found {len(signals)} Wick Sweep Long patterns in neutral conditions")
        
        # Execute trades
        equity = CAPITAL
        equity_curve = [CAPITAL]
        
        logger.info("\n" + "=" * 70)
        logger.info("💼 EXECUTING FOCUSED STRATEGY")
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
        """Print comprehensive results."""
        logger.info("\n" + "=" * 70)
        logger.info("📊 V4 RESULTS - FOCUSED NEUTRAL STRUCTURE STRATEGY")
        logger.info("=" * 70)
        
        logger.info(f"\n💰 CAPITAL")
        logger.info(f"   Starting:     ${performance['capital']['starting']:,.2f}")
        logger.info(f"   Ending:       ${performance['capital']['ending']:,.2f}")
        logger.info(f"   Return:       {performance['capital']['total_return']*100:.2f}%")
        logger.info(f"   Profit:       ${performance['capital']['total_pnl']:,.2f}")
        
        logger.info(f"\n📈 TRADES")
        logger.info(f"   Total:        {performance['trades']['total']}")
        logger.info(f"   Winners:      {performance['trades']['winners']}")
        logger.info(f"   Losers:       {performance['trades']['losers']}")
        logger.info(f"   Win Rate:     {performance['performance']['win_rate']*100:.1f}%")
        logger.info(f"   Profit Factor: {performance['performance']['profit_factor']:.2f}")
        logger.info(f"   Expectancy:   ${performance['performance']['expectancy']:.2f}")
        
        logger.info(f"\n💵 WIN/LOSS")
        logger.info(f"   Avg Win:      ${performance['performance']['avg_win']:.2f}")
        logger.info(f"   Avg Loss:     ${performance['performance']['avg_loss']:.2f}")
        logger.info(f"   Avg Cost:     ${performance['risk']['avg_transaction_cost']:.2f}")
        
        logger.info(f"\n⚠️  RISK")
        logger.info(f"   Max DD:       {performance['risk']['max_drawdown']*100:.2f}%")
        logger.info(f"   Sharpe:       {performance['risk']['sharpe_ratio']:.2f}")
        
        if mc_results:
            logger.info(f"\n🎲 MONTE CARLO ({MONTE_CARLO_RUNS:,} runs)")
            logger.info(f"   Median:       ${mc_results['final_equity']['median']:,.2f}")
            logger.info(f"   5th %:        ${mc_results['final_equity']['percentile_5']:,.2f}")
            logger.info(f"   95th %:       ${mc_results['final_equity']['percentile_95']:,.2f}")
            logger.info(f"   Expected DD:  {mc_results['max_drawdown']['median']*100:.2f}%")
        
        if mae_mfe:
            logger.info(f"\n📉 MAE/MFE")
            logger.info(f"   Winners: MAE ${mae_mfe['winners']['avg_mae']:.2f} | MFE ${mae_mfe['winners']['avg_mfe']:.2f}")
            logger.info(f"   Losers:  MAE ${mae_mfe['losers']['avg_mae']:.2f} | MFE ${mae_mfe['losers']['avg_mfe']:.2f}")
        
        logger.info("\n" + "=" * 70)
        
        # V4 specific insights
        if self.structure_analysis:
            neutral_count = sum(1 for s in self.structure_analysis if s['is_neutral'])
            total_days = len(self.structure_analysis)
            neutral_pct = (neutral_count / total_days * 100) if total_days > 0 else 0
            
            logger.info(f"📊 STRUCTURE ANALYSIS")
            logger.info(f"   Trading Days Analyzed: {total_days}")
            logger.info(f"   Neutral Days Found:    {neutral_count} ({neutral_pct:.1f}%)")
            logger.info(f"   Directional Days:      {total_days - neutral_count} ({100-neutral_pct:.1f}%)")
        
        logger.info("\n" + "=" * 70)
        logger.info("✅ V4 BACKTEST COMPLETE")
        logger.info("=" * 70)


if __name__ == "__main__":
    backtest = MokshaV4NeutralStructure()
    results = backtest.run()
    
    if results:
        output_file = f"/mnt/user-data/outputs/moksha_v4_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        serializable_results = {
            'performance': results['performance'],
            'monte_carlo': results['monte_carlo'],
            'mae_mfe': results['mae_mfe'],
            'equity_curve': results['equity_curve'],
            'total_trades': len(results['trade_history']),
            'neutral_days': sum(1 for s in results['structure_analysis'] if s['is_neutral']),
            'total_days': len(results['structure_analysis']),
            'config': {
                'symbol': SYMBOL,
                'capital': CAPITAL,
                'pattern': 'Wick Sweep Long Only',
                'structure': 'Neutral Only',
                'time_window': f'{Q2_START_TIME}-{Q2_END_TIME}',
                'profit_target': f'{TAKE_PROFIT_RATIO}R',
                'breakeven': f'{BREAKEVEN_RATIO}R',
                'neutral_range': f'{NEUTRAL_RANGE_MIN}-{NEUTRAL_RANGE_MAX}',
                'neutral_momentum': f'<{NEUTRAL_MOMENTUM_MAX*100}%',
                'range_filter': f'<{MAX_Q1_RANGE_VS_ATR}x ATR'
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"\n💾 Results saved: {output_file}")