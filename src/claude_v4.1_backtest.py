"""
MOKSHA CAPITAL - STRATEGY V4.1: RELAXED NEUTRAL CRITERIA
Learning from V4: Finding the Balance Between Too Strict and Too Loose

V4 PROBLEM:
-----------
Found ZERO neutral days out of 152 trading days.
Filters were too restrictive when combined:
- Range position 0.35-0.65 (middle 30%)
- Momentum < 0.2%
- Range < 1.0x ATR

These three conditions together passed 0% of days (!)

V4.1 SOLUTION:
--------------
Relax each filter to find middle ground:
- Range position 0.30-0.70 (middle 40% instead of 30%)
- Momentum < 0.4% (instead of 0.2%)
- Range < 1.5x ATR (instead of 1.0x, or make optional)

Goal: Find 20-40 neutral days (15-25% of total)
This should give us 15-30 actual trades

DIAGNOSTIC MODE:
----------------
V4.1 includes detailed logging showing:
- How many days pass EACH individual filter
- Which filter is most restrictive
- What combination works best

This helps us understand the tradeoff between selectivity and sample size.

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
# CONFIGURATION - RELAXED FOR V4.1
# ============================================================================

SYMBOL = 'AGQ'
START_DATE = '2024-01-01'
CAPITAL = 25000.0
BASE_RISK_PCT = 0.015
MAX_RISK_PCT = 0.025
MIN_POSITION_SIZE = 10

TRADE_ONLY_WICK_SWEEP_LONG = True

# ATR and Risk
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 1.5
TAKE_PROFIT_RATIO = 1.25
BREAKEVEN_RATIO = 0.75

# Transaction Costs
SLIPPAGE_BPS = 1.5
SPREAD_DOLLARS = 0.02
COMMISSION = 0.0

# V4.1: RELAXED NEUTRAL STRUCTURE CRITERIA
USE_NEUTRAL_STRUCTURE_FILTER = True

# RELAXED: Widened from 0.35-0.65 to 0.30-0.70
NEUTRAL_RANGE_MIN = 0.30  # Was 0.35
NEUTRAL_RANGE_MAX = 0.70  # Was 0.65
# Now accepting middle 40% instead of middle 30%

# RELAXED: Increased from 0.002 (0.2%) to 0.004 (0.4%)
NEUTRAL_MOMENTUM_MAX = 0.004  # Was 0.002
# Allows more movement during Q1 while still being "choppy"

# RELAXED: Made optional and increased threshold
REQUIRE_RANGE_CONTRACTION = False  # Was True - TRY WITHOUT THIS FILTER FIRST
MAX_Q1_RANGE_VS_ATR = 1.5  # Was 1.0 - if we enable it, use looser threshold

# DIAGNOSTIC MODE: Show detailed filter analysis
DIAGNOSTIC_MODE = True  # Set to True to see which filters are most restrictive

# Time Window (keep focused)
TRADE_ONLY_EARLY_Q2 = True
Q2_START_TIME = "10:00"
Q2_END_TIME = "10:30"

# Signal Quality
MIN_VOLUME_MULTIPLIER = 1.2
MIN_Q1_BARS = 15
MAX_DAILY_TRADES = 2

# Risk Management
MAX_CONSECUTIVE_LOSSES = 2
POSITION_SCALE_DOWN = 0.50
MAX_DAILY_LOSS_PCT = 0.04
KELLY_FRACTION = 0.25

# Statistical
MONTE_CARLO_RUNS = 5000
MIN_TRADES_FOR_KELLY = 15

# ============================================================================
# MOKSHA V4.1: RELAXED NEUTRAL STRUCTURE ENGINE
# ============================================================================

class MokshaV41RelaxedNeutral:
    """
    Version 4.1: Relaxed neutral structure criteria.
    
    Goal: Find the sweet spot between V3 (88 trades, too many) 
    and V4 (0 trades, too few).
    
    Target: 20-40 neutral days → 15-30 actual trades
    """
    
    def __init__(self):
        """Initialize with diagnostic tracking."""
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
        
        # NEW: Diagnostic tracking
        self.diagnostic_data = {
            'total_days': 0,
            'pass_range_position': 0,
            'pass_momentum': 0,
            'pass_range_vs_atr': 0,
            'pass_all_filters': 0,
            'range_positions': [],
            'momentums': [],
            'range_vs_atrs': []
        }
        
    def calculate_transaction_costs(self, shares, entry_price):
        """Calculate transaction costs."""
        position_value = shares * entry_price
        slippage_cost = (position_value * SLIPPAGE_BPS / 10000) * 2
        spread_cost = shares * SPREAD_DOLLARS * 2
        commission_cost = COMMISSION * 2
        return slippage_cost + spread_cost + commission_cost
    
    def calculate_atr(self, df, period=ATR_PERIOD):
        """Calculate ATR."""
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
        Analyze Q1 structure with RELAXED neutral criteria.
        
        V4.1 Changes:
        - Wider range acceptance (30-70% instead of 35-65%)
        - Higher momentum tolerance (0.4% instead of 0.2%)
        - Optional range contraction filter (or looser if enabled)
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
        
        # Calculate metrics
        range_position = (q1_close - q1_low) / q1_range
        q1_momentum = (q1_close - q1_open) / q1_open
        q1_atr = q1_data['atr'].iloc[-1] if not pd.isna(q1_data['atr'].iloc[-1]) else q1_range * 0.5
        range_vs_atr = q1_range / q1_atr if q1_atr > 0 else 1.0
        q1_volume = q1_data['volume'].sum()
        q1_avg_volume = q1_volume / len(q1_data)
        
        # Diagnostic tracking
        if DIAGNOSTIC_MODE:
            self.diagnostic_data['total_days'] += 1
            self.diagnostic_data['range_positions'].append(range_position)
            self.diagnostic_data['momentums'].append(abs(q1_momentum))
            self.diagnostic_data['range_vs_atrs'].append(range_vs_atr)
        
        # ====================================================================
        # V4.1 RELAXED NEUTRAL CRITERIA
        # ====================================================================
        
        # Filter 1: Range position (RELAXED)
        in_neutral_range = (range_position >= NEUTRAL_RANGE_MIN and 
                          range_position <= NEUTRAL_RANGE_MAX)
        
        if DIAGNOSTIC_MODE and in_neutral_range:
            self.diagnostic_data['pass_range_position'] += 1
        
        # Filter 2: Low momentum (RELAXED)
        low_momentum = abs(q1_momentum) < NEUTRAL_MOMENTUM_MAX
        
        if DIAGNOSTIC_MODE and low_momentum:
            self.diagnostic_data['pass_momentum'] += 1
        
        # Filter 3: Range contraction (OPTIONAL/RELAXED)
        if REQUIRE_RANGE_CONTRACTION:
            range_ok = range_vs_atr < MAX_Q1_RANGE_VS_ATR
        else:
            range_ok = True  # Always pass if filter disabled
        
        if DIAGNOSTIC_MODE and range_ok:
            self.diagnostic_data['pass_range_vs_atr'] += 1
        
        # Pass if ALL enabled filters pass
        is_neutral = in_neutral_range and low_momentum and range_ok
        
        if DIAGNOSTIC_MODE and is_neutral:
            self.diagnostic_data['pass_all_filters'] += 1
        
        # Classify structure type
        if in_neutral_range and low_momentum:
            structure_type = 'neutral'
        elif range_position > 0.70 and q1_momentum > 0.004:
            structure_type = 'bullish'
        elif range_position < 0.30 and q1_momentum < -0.004:
            structure_type = 'bearish'
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
    
    def print_diagnostic_summary(self):
        """
        Print detailed diagnostic information about filter selectivity.
        
        This helps understand which filters are most restrictive and
        whether we need to relax further or tighten.
        """
        if not DIAGNOSTIC_MODE or self.diagnostic_data['total_days'] == 0:
            return
        
        total = self.diagnostic_data['total_days']
        
        logger.info("\n" + "=" * 70)
        logger.info("🔬 DIAGNOSTIC ANALYSIS: FILTER SELECTIVITY")
        logger.info("=" * 70)
        
        logger.info(f"\nTotal Trading Days Analyzed: {total}")
        
        # Individual filter pass rates
        range_pass = self.diagnostic_data['pass_range_position']
        momentum_pass = self.diagnostic_data['pass_momentum']
        atr_pass = self.diagnostic_data['pass_range_vs_atr']
        all_pass = self.diagnostic_data['pass_all_filters']
        
        logger.info(f"\nIndividual Filter Pass Rates:")
        logger.info(f"  Range Position (0.30-0.70):     {range_pass}/{total} ({range_pass/total*100:.1f}%)")
        logger.info(f"  Low Momentum (<0.4%):            {momentum_pass}/{total} ({momentum_pass/total*100:.1f}%)")
        if REQUIRE_RANGE_CONTRACTION:
            logger.info(f"  Range Contraction (<1.5x ATR):   {atr_pass}/{total} ({atr_pass/total*100:.1f}%)")
        else:
            logger.info(f"  Range Contraction:               DISABLED")
        
        logger.info(f"\nCombined (All Filters):            {all_pass}/{total} ({all_pass/total*100:.1f}%)")
        
        # Show distributions
        if self.diagnostic_data['range_positions']:
            rps = self.diagnostic_data['range_positions']
            logger.info(f"\nRange Position Distribution:")
            logger.info(f"  Min:  {min(rps):.3f}")
            logger.info(f"  25th: {np.percentile(rps, 25):.3f}")
            logger.info(f"  50th: {np.percentile(rps, 50):.3f}")
            logger.info(f"  75th: {np.percentile(rps, 75):.3f}")
            logger.info(f"  Max:  {max(rps):.3f}")
        
        if self.diagnostic_data['momentums']:
            moms = self.diagnostic_data['momentums']
            logger.info(f"\nMomentum Distribution (absolute %):")
            logger.info(f"  Min:  {min(moms)*100:.3f}%")
            logger.info(f"  25th: {np.percentile(moms, 25)*100:.3f}%")
            logger.info(f"  50th: {np.percentile(moms, 50)*100:.3f}%")
            logger.info(f"  75th: {np.percentile(moms, 75)*100:.3f}%")
            logger.info(f"  Max:  {max(moms)*100:.3f}%")
        
        if self.diagnostic_data['range_vs_atrs']:
            rvs = self.diagnostic_data['range_vs_atrs']
            logger.info(f"\nRange vs ATR Distribution:")
            logger.info(f"  Min:  {min(rvs):.3f}x")
            logger.info(f"  25th: {np.percentile(rvs, 25):.3f}x")
            logger.info(f"  50th: {np.percentile(rvs, 50):.3f}x")
            logger.info(f"  75th: {np.percentile(rvs, 75):.3f}x")
            logger.info(f"  Max:  {max(rvs):.3f}x")
        
        # Recommendations
        logger.info(f"\n💡 OPTIMIZATION SUGGESTIONS:")
        
        if all_pass < 15:
            logger.info(f"  ⚠️  Only {all_pass} neutral days found - still too restrictive")
            logger.info(f"  Consider:")
            if range_pass / total < 0.30:
                logger.info(f"    - Widen range criteria further (try 0.25-0.75)")
            if momentum_pass / total < 0.30:
                logger.info(f"    - Increase momentum threshold (try 0.6% or 0.8%)")
            if REQUIRE_RANGE_CONTRACTION and atr_pass / total < 0.30:
                logger.info(f"    - Disable range contraction filter entirely")
        
        elif all_pass > 60:
            logger.info(f"  ✅ Found {all_pass} neutral days - might be too loose")
            logger.info(f"  Consider tightening if you want higher selectivity")
        
        else:
            logger.info(f"  ✅ Found {all_pass} neutral days - good balance!")
            logger.info(f"  This should generate 15-40 actual trades")
        
        logger.info("=" * 70)
    
    def fetch_real_data(self):
        """Fetch data."""
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
        Pattern identification with relaxed neutral structure filter.
        """
        df['time_only'] = df.index.time
        t_open = pd.Timestamp("08:30").time()
        t_q1_end = pd.Timestamp("10:00").time()
        t_q2_start = pd.Timestamp(Q2_START_TIME).time()
        t_q2_end = pd.Timestamp(Q2_END_TIME).time()
        
        signals = []
        grouped = df.groupby(df.index.date)
        
        logger.info(f"\n{'='*70}")
        logger.info(f"🎯 V4.1: RELAXED NEUTRAL STRUCTURE")
        logger.info(f"{'='*70}")
        logger.info(f"Range Position: {NEUTRAL_RANGE_MIN:.2f}-{NEUTRAL_RANGE_MAX:.2f} (middle {(NEUTRAL_RANGE_MAX-NEUTRAL_RANGE_MIN)*100:.0f}%)")
        logger.info(f"Momentum Threshold: <{NEUTRAL_MOMENTUM_MAX*100:.1f}%")
        logger.info(f"Range Contraction: {'ENABLED' if REQUIRE_RANGE_CONTRACTION else 'DISABLED'}")
        if REQUIRE_RANGE_CONTRACTION:
            logger.info(f"  Max Range vs ATR: {MAX_Q1_RANGE_VS_ATR}x")
        logger.info(f"{'='*70}\n")
        
        neutral_days_count = 0
        directional_days_count = 0
        
        for date, day_data in tqdm(grouped, desc="🔬 Scanning with Relaxed Criteria"):
            if len(day_data) < 30:
                continue
            
            if self.check_daily_loss_limit(pd.Timestamp(date)):
                continue
            
            daily_trades = len([s for s in signals if s['time'].date() == date])
            if daily_trades >= MAX_DAILY_TRADES:
                continue
            
            # Q1 Analysis
            q1_data = day_data[
                (day_data['time_only'] >= t_open) & 
                (day_data['time_only'] < t_q1_end)
            ]
            
            if len(q1_data) < MIN_Q1_BARS:
                continue
            
            # Check if neutral with RELAXED criteria
            is_neutral, q1_metrics = self.analyze_q1_structure(q1_data)
            
            self.structure_analysis.append({
                'date': date,
                'is_neutral': is_neutral,
                **q1_metrics
            })
            
            if not is_neutral:
                directional_days_count += 1
                self.filtered_signals.append({
                    'date': date,
                    'reason': f"not_neutral_{q1_metrics['structure_type']}",
                    'range_position': q1_metrics['range_position'],
                    'momentum': q1_metrics['momentum_pct'],
                    'range_vs_atr': q1_metrics['range_vs_atr'],
                    'failed_filters': {
                        'range': not q1_metrics['in_neutral_range'],
                        'momentum': not q1_metrics['low_momentum'],
                        'range_vs_atr': not q1_metrics['range_ok']
                    }
                })
                continue
            
            neutral_days_count += 1
            
            # Q2 Pattern Scanning
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
                
                # Wick Sweep Long pattern
                if curr['low'] < q1_low and curr['close'] > q1_low:
                    signal = 1
                    setup_type = "Wick Sweep Long"
                    stop_loss = curr['low'] - (current_atr * ATR_STOP_MULTIPLIER)
                    
                    if curr['volume'] < q1_avg_volume * MIN_VOLUME_MULTIPLIER:
                        confidence = 0.85
                    else:
                        confidence = 1.0
                    
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
                    
                    break
        
        # Print diagnostic summary
        self.print_diagnostic_summary()
        
        logger.info(f"\n✅ Scan Complete:")
        logger.info(f"   Neutral Days Found: {neutral_days_count}")
        logger.info(f"   Directional Days Skipped: {directional_days_count}")
        logger.info(f"   Signals Found: {len(signals)}")
        
        return pd.DataFrame(signals)
    
    def check_daily_loss_limit(self, current_date):
        """Check daily loss limit."""
        date_key = current_date.date()
        daily_pnl = self.daily_pnl.get(date_key, 0.0)
        return daily_pnl < -CAPITAL * MAX_DAILY_LOSS_PCT
    
    def calculate_kelly_position_size(self):
        """Kelly position sizing."""
        if len(self.trade_history) < MIN_TRADES_FOR_KELLY:
            return BASE_RISK_PCT
        
        recent_trades = self.trade_history[-30:]
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
        """Execute trade."""
        entry = setup['entry']
        stop = setup['stop']
        direction = setup['signal']
        setup_type = setup['type']
        confidence = setup['confidence']
        
        kelly_risk = self.calculate_kelly_position_size()
        adjusted_risk = kelly_risk * confidence * self.current_position_scale
        
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
        
        future_data = df[df.index > setup['time']].head(48)
        
        outcome = 0
        stop_moved_to_breakeven = False
        exit_price = entry
        exit_reason = "time"
        
        max_favorable = 0
        max_adverse = 0
        
        for idx, bar in future_data.iterrows():
            current_stop = entry if stop_moved_to_breakeven else stop
            
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
        
        if outcome == 0 and not future_data.empty:
            exit_price = future_data.iloc[-1]['close']
            if stop_moved_to_breakeven and exit_price < entry:
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
        if len(self.trade_history) < 15:
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
        logger.info("🚀 MOKSHA V4.1 - RELAXED NEUTRAL STRUCTURE")
        logger.info("=" * 70)
        logger.info("Finding the Balance: Not Too Strict, Not Too Loose")
        logger.info("=" * 70)
        
        df = self.fetch_real_data()
        if df.empty:
            return None
        
        signals = self.identify_wick_sweep_long_patterns(df)
        
        if signals.empty:
            logger.error("❌ Still no signals found")
            logger.info("\n💡 Filters may still be too strict. Check diagnostic output above.")
            return None
        
        logger.info(f"✅ Found {len(signals)} signals in neutral conditions")
        
        equity = CAPITAL
        equity_curve = [CAPITAL]
        
        logger.info("\n" + "=" * 70)
        logger.info("💼 EXECUTING TRADES")
        logger.info("=" * 70)
        
        for idx, setup in tqdm(signals.iterrows(), total=len(signals), desc="⚡ Trading"):
            pnl, trade_record = self.execute_trade(setup, df, equity)
            equity += pnl
            equity_curve.append(equity)
        
        performance = self.generate_performance_report(equity_curve, signals)
        mc_results = self.monte_carlo_simulation()
        mae_mfe = self.analyze_mae_mfe()
        
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
        logger.info("📊 V4.1 RESULTS")
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
        
        logger.info(f"\n⚠️  RISK")
        logger.info(f"   Max DD:       {performance['risk']['max_drawdown']*100:.2f}%")
        logger.info(f"   Sharpe:       {performance['risk']['sharpe_ratio']:.2f}")
        
        if mc_results:
            logger.info(f"\n🎲 MONTE CARLO")
            logger.info(f"   Median:       ${mc_results['final_equity']['median']:,.2f}")
            logger.info(f"   5th %:        ${mc_results['final_equity']['percentile_5']:,.2f}")
            logger.info(f"   95th %:       ${mc_results['final_equity']['percentile_95']:,.2f}")
        
        if mae_mfe:
            logger.info(f"\n📉 MAE/MFE")
            logger.info(f"   Winners: MAE ${mae_mfe['winners']['avg_mae']:.2f} | MFE ${mae_mfe['winners']['avg_mfe']:.2f}")
            logger.info(f"   Losers:  MAE ${mae_mfe['losers']['avg_mae']:.2f} | MFE ${mae_mfe['losers']['avg_mfe']:.2f}")
        
        logger.info("\n" + "=" * 70)
        logger.info("✅ V4.1 BACKTEST COMPLETE")
        logger.info("=" * 70)


if __name__ == "__main__":
    backtest = MokshaV41RelaxedNeutral()
    results = backtest.run()
    
    if results:
        
        output_file = f"/Users/viveksingh/Documents/Patience/Analysis/TARS/MOKSHA/Moksha_1/moksha_v4.1_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
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
                'neutral_range': f'{NEUTRAL_RANGE_MIN}-{NEUTRAL_RANGE_MAX}',
                'neutral_momentum': f'<{NEUTRAL_MOMENTUM_MAX*100}%',
                'range_contraction': 'DISABLED' if not REQUIRE_RANGE_CONTRACTION else f'<{MAX_Q1_RANGE_VS_ATR}x ATR',
                'time_window': f'{Q2_START_TIME}-{Q2_END_TIME}',
                'profit_target': f'{TAKE_PROFIT_RATIO}R'
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"\n💾 Results saved: {output_file}")