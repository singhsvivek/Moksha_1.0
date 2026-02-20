"""
MOKSHA CAPITAL - STRATEGY V4.2: THE PRAGMATIC APPROACH
Translating Four Versions of Learning Into Actual Profitability

THE JOURNEY TO V4.2:
--------------------
V1: Lost $79 trading everything (101 trades, 37.6% WR)
    → Learned: Wick Sweep Long works (54% WR), shorts don't

V2: Lost $2,319 with daily trend filter (2 trades, 0% WR)  
    → Learned: Wrong timescale destroys edge

V3: Lost $1,481 with intraday structure (88 trades, 40.9% WR)
    → Learned: NEUTRAL structure profitable (+$1,215), directional loses

V4: No trades - too restrictive (0 trades)
    → Learned: Perfect is the enemy of good

V4.1: Lost $388 - still too strict (1 trade, 0% WR, only 11 neutral days)
    → Learned: 2024-2025 period was not very choppy

V4.2 PHILOSOPHY:
----------------
We now know:
1. Wick Sweep Long has demonstrated edge (54% WR in V1, V3)
2. Neutral structure outperforms directional (V3 proved this)
3. Too much filtering eliminates all opportunities (V4, V4.1)

So V4.2 takes a PRAGMATIC approach:
- Wide enough filters to get 20-40 trades (adequate sample)
- Selective enough to avoid the worst conditions
- Focus on the one pattern that consistently works
- Accept "neutral enough" instead of demanding perfect neutrality

GOAL: Turn learning into profit, not perfection into paralysis.

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
# CONFIGURATION - PRAGMATIC SETTINGS FOR REAL RESULTS
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
TAKE_PROFIT_RATIO = 1.25  # Might test 1.3 or 1.4 if we're getting too many breakevens
BREAKEVEN_RATIO = 0.75

# Transaction Costs (realistic based on V3 actual avg of $23)
SLIPPAGE_BPS = 1.5
SPREAD_DOLLARS = 0.02
COMMISSION = 0.0

# ============================================================================
# V4.2: PRAGMATIC NEUTRAL STRUCTURE CRITERIA
# ============================================================================
# These are SIGNIFICANTLY relaxed to ensure adequate sample size

USE_NEUTRAL_STRUCTURE_FILTER = True

# OPTION 1: Relaxed Range + Momentum (DEFAULT)
NEUTRAL_RANGE_MIN = 0.25  # Was 0.30 in V4.1
NEUTRAL_RANGE_MAX = 0.75  # Was 0.70 in V4.1
# Now accepts middle 50% instead of middle 40%

NEUTRAL_MOMENTUM_MAX = 0.008  # Was 0.004 (0.4%) in V4.1
# Now 0.8% - much more tolerant of directional movement
# On a $34 stock, this allows ~$0.27 of Q1 movement

REQUIRE_RANGE_CONTRACTION = False
MAX_Q1_RANGE_VS_ATR = 1.5

# OPTION 2: Momentum-Only Mode (ALTERNATIVE TEST)
# Set USE_MOMENTUM_ONLY_FILTER = True to test this simpler approach
USE_MOMENTUM_ONLY_FILTER = False  # Set True to ignore range position entirely
MOMENTUM_ONLY_THRESHOLD = 0.008   # Same 0.8% threshold

# Time Window - EXTENDED back to 60 minutes for more opportunities
TRADE_ONLY_EARLY_Q2 = True
Q2_START_TIME = "10:00"
Q2_END_TIME = "11:00"  # Back to 60-minute window (was 30 min in V4/V4.1)

# Signal Quality
MIN_VOLUME_MULTIPLIER = 1.2
MIN_Q1_BARS = 15
MAX_DAILY_TRADES = 3  # Increased from 2 to allow more opportunities

# Risk Management
MAX_CONSECUTIVE_LOSSES = 2
POSITION_SCALE_DOWN = 0.50
MAX_DAILY_LOSS_PCT = 0.04
KELLY_FRACTION = 0.25

# Diagnostic and Analysis
DIAGNOSTIC_MODE = True
MONTE_CARLO_RUNS = 5000
MIN_TRADES_FOR_KELLY = 15

# ============================================================================
# V4.2: PRAGMATIC BACKTEST ENGINE
# ============================================================================

class MokshaV42Pragmatic:
    """
    Version 4.2: The pragmatic approach to neutral structure trading.
    
    Philosophy: Better to have 25 trades at 52% WR than 0 trades at
    "perfect" conditions. Trade in neutral-enough conditions.
    """
    
    def __init__(self):
        """Initialize with comprehensive tracking."""
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
        
        # Enhanced diagnostic tracking
        self.diagnostic_data = {
            'total_days': 0,
            'pass_range_position': 0,
            'pass_momentum': 0,
            'pass_both_filters': 0,
            'pass_momentum_only': 0,  # For momentum-only comparison
            'range_positions': [],
            'momentums': [],
            'range_vs_atrs': [],
            'by_threshold': {}  # Track pass rates at different threshold levels
        }
        
    def calculate_transaction_costs(self, shares, entry_price):
        """Calculate realistic transaction costs based on V3 actuals."""
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
        Analyze Q1 structure with PRAGMATIC neutral criteria.
        
        V4.2 accepts "neutral enough" conditions:
        - Middle 50% of range (25-75%)
        - Less than 0.8% momentum
        - No range contraction requirement
        
        Also tracks what would happen with different thresholds.
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
        
        # Calculate all metrics
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
            
            # Track pass rates at different threshold levels for analysis
            # This helps us see what would have happened with tighter/looser filters
            test_range_thresholds = [(0.30, 0.70), (0.25, 0.75), (0.20, 0.80)]
            test_momentum_thresholds = [0.004, 0.006, 0.008, 0.010]
            
            for rmin, rmax in test_range_thresholds:
                key = f"range_{rmin}_{rmax}"
                if key not in self.diagnostic_data['by_threshold']:
                    self.diagnostic_data['by_threshold'][key] = 0
                if rmin <= range_position <= rmax:
                    self.diagnostic_data['by_threshold'][key] += 1
            
            for mom_thresh in test_momentum_thresholds:
                key = f"momentum_{mom_thresh}"
                if key not in self.diagnostic_data['by_threshold']:
                    self.diagnostic_data['by_threshold'][key] = 0
                if abs(q1_momentum) < mom_thresh:
                    self.diagnostic_data['by_threshold'][key] += 1
        
        # ====================================================================
        # V4.2 FILTER LOGIC
        # ====================================================================
        
        if USE_MOMENTUM_ONLY_FILTER:
            # SIMPLE MODE: Only check momentum, ignore range position
            is_neutral = abs(q1_momentum) < MOMENTUM_ONLY_THRESHOLD
            
            if DIAGNOSTIC_MODE and is_neutral:
                self.diagnostic_data['pass_momentum_only'] += 1
            
            structure_type = 'neutral_momentum' if is_neutral else 'directional_momentum'
            
        else:
            # DEFAULT MODE: Range position + Momentum
            in_neutral_range = (range_position >= NEUTRAL_RANGE_MIN and 
                              range_position <= NEUTRAL_RANGE_MAX)
            
            low_momentum = abs(q1_momentum) < NEUTRAL_MOMENTUM_MAX
            
            if REQUIRE_RANGE_CONTRACTION:
                range_ok = range_vs_atr < MAX_Q1_RANGE_VS_ATR
            else:
                range_ok = True
            
            # Diagnostic tracking for each filter
            if DIAGNOSTIC_MODE:
                if in_neutral_range:
                    self.diagnostic_data['pass_range_position'] += 1
                if low_momentum:
                    self.diagnostic_data['pass_momentum'] += 1
                if in_neutral_range and low_momentum:
                    self.diagnostic_data['pass_both_filters'] += 1
            
            is_neutral = in_neutral_range and low_momentum and range_ok
            
            # Classify structure type for tracking
            if in_neutral_range and low_momentum:
                structure_type = 'neutral'
            elif range_position > 0.75 and q1_momentum > 0.008:
                structure_type = 'bullish'
            elif range_position < 0.25 and q1_momentum < -0.008:
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
            'is_neutral': is_neutral
        }
        
        return is_neutral, metrics
    
    def print_comprehensive_diagnostics(self):
        """
        Print detailed diagnostic analysis showing filter effectiveness
        at multiple threshold levels.
        
        This helps understand tradeoffs between selectivity and sample size.
        """
        if not DIAGNOSTIC_MODE or self.diagnostic_data['total_days'] == 0:
            return
        
        total = self.diagnostic_data['total_days']
        
        logger.info("\n" + "=" * 80)
        logger.info("🔬 COMPREHENSIVE DIAGNOSTIC ANALYSIS")
        logger.info("=" * 80)
        
        logger.info(f"\nTotal Trading Days Analyzed: {total}")
        logger.info(f"Date Range: {START_DATE} to present")
        
        # Current configuration results
        logger.info(f"\n📊 CURRENT CONFIGURATION RESULTS:")
        logger.info(f"   Filter Mode: {'MOMENTUM ONLY' if USE_MOMENTUM_ONLY_FILTER else 'RANGE + MOMENTUM'}")
        
        if USE_MOMENTUM_ONLY_FILTER:
            mom_pass = self.diagnostic_data['pass_momentum_only']
            logger.info(f"   Momentum (<{MOMENTUM_ONLY_THRESHOLD*100:.1f}%): {mom_pass}/{total} ({mom_pass/total*100:.1f}%)")
            logger.info(f"   → {mom_pass} neutral days found")
        else:
            range_pass = self.diagnostic_data['pass_range_position']
            mom_pass = self.diagnostic_data['pass_momentum']
            both_pass = self.diagnostic_data['pass_both_filters']
            
            logger.info(f"   Range Position ({NEUTRAL_RANGE_MIN:.2f}-{NEUTRAL_RANGE_MAX:.2f}): {range_pass}/{total} ({range_pass/total*100:.1f}%)")
            logger.info(f"   Momentum (<{NEUTRAL_MOMENTUM_MAX*100:.1f}%): {mom_pass}/{total} ({mom_pass/total*100:.1f}%)")
            logger.info(f"   BOTH Filters Pass: {both_pass}/{total} ({both_pass/total*100:.1f}%)")
            logger.info(f"   → {both_pass} neutral days found")
        
        # Alternative threshold analysis
        logger.info(f"\n🔍 SENSITIVITY ANALYSIS (What if we used different thresholds?):")
        logger.info(f"\n   Range Position Thresholds:")
        for rmin, rmax in [(0.20, 0.80), (0.25, 0.75), (0.30, 0.70), (0.35, 0.65)]:
            key = f"range_{rmin}_{rmax}"
            count = self.diagnostic_data['by_threshold'].get(key, 0)
            pct = (count / total * 100) if total > 0 else 0
            marker = "← CURRENT" if (rmin == NEUTRAL_RANGE_MIN and rmax == NEUTRAL_RANGE_MAX) else ""
            logger.info(f"      {rmin:.2f}-{rmax:.2f} (middle {int((rmax-rmin)*100)}%): {count} days ({pct:.1f}%) {marker}")
        
        logger.info(f"\n   Momentum Thresholds:")
        for mom_thresh in [0.004, 0.006, 0.008, 0.010, 0.012]:
            key = f"momentum_{mom_thresh}"
            count = self.diagnostic_data['by_threshold'].get(key, 0)
            pct = (count / total * 100) if total > 0 else 0
            marker = "← CURRENT" if mom_thresh == NEUTRAL_MOMENTUM_MAX else ""
            logger.info(f"      <{mom_thresh*100:.1f}%: {count} days ({pct:.1f}%) {marker}")
        
        # Distribution statistics
        if self.diagnostic_data['range_positions']:
            rps = self.diagnostic_data['range_positions']
            logger.info(f"\n📈 RANGE POSITION DISTRIBUTION:")
            logger.info(f"   Min:    {min(rps):.3f} (close at very low)")
            logger.info(f"   25th %: {np.percentile(rps, 25):.3f}")
            logger.info(f"   Median: {np.percentile(rps, 50):.3f} (typical day)")
            logger.info(f"   75th %: {np.percentile(rps, 75):.3f}")
            logger.info(f"   Max:    {max(rps):.3f} (close at very high)")
            
            # Count how many days in each tercile
            low_count = sum(1 for rp in rps if rp < 0.33)
            mid_count = sum(1 for rp in rps if 0.33 <= rp <= 0.67)
            high_count = sum(1 for rp in rps if rp > 0.67)
            logger.info(f"\n   Distribution by tercile:")
            logger.info(f"      Low (0-33%):   {low_count} days ({low_count/total*100:.1f}%)")
            logger.info(f"      Mid (33-67%):  {mid_count} days ({mid_count/total*100:.1f}%)")
            logger.info(f"      High (67-100%): {high_count} days ({high_count/total*100:.1f}%)")
        
        if self.diagnostic_data['momentums']:
            moms = self.diagnostic_data['momentums']
            logger.info(f"\n📈 MOMENTUM DISTRIBUTION (absolute %):")
            logger.info(f"   Min:    {min(moms)*100:.3f}%")
            logger.info(f"   25th %: {np.percentile(moms, 25)*100:.3f}%")
            logger.info(f"   Median: {np.percentile(moms, 50)*100:.3f}% (typical day)")
            logger.info(f"   75th %: {np.percentile(moms, 75)*100:.3f}%")
            logger.info(f"   Max:    {max(moms)*100:.3f}%")
            
            # Categorize days by momentum strength
            very_low = sum(1 for m in moms if m < 0.002)
            low = sum(1 for m in moms if 0.002 <= m < 0.005)
            moderate = sum(1 for m in moms if 0.005 <= m < 0.010)
            high = sum(1 for m in moms if m >= 0.010)
            
            logger.info(f"\n   Distribution by momentum:")
            logger.info(f"      Very Low (<0.2%):  {very_low} days ({very_low/total*100:.1f}%)")
            logger.info(f"      Low (0.2-0.5%):     {low} days ({low/total*100:.1f}%)")
            logger.info(f"      Moderate (0.5-1.0%): {moderate} days ({moderate/total*100:.1f}%)")
            logger.info(f"      High (>1.0%):       {high} days ({high/total*100:.1f}%)")
        
        # Recommendations
        neutral_count = self.diagnostic_data['pass_both_filters'] if not USE_MOMENTUM_ONLY_FILTER else self.diagnostic_data['pass_momentum_only']
        
        logger.info(f"\n💡 RECOMMENDATIONS:")
        if neutral_count < 20:
            logger.info(f"   ⚠️  Only {neutral_count} neutral days - STILL TOO RESTRICTIVE")
            logger.info(f"   Suggestions:")
            logger.info(f"      → Try MOMENTUM_ONLY mode (set USE_MOMENTUM_ONLY_FILTER = True)")
            logger.info(f"      → Or widen range to 0.20-0.80 (middle 60%)")
            logger.info(f"      → Or increase momentum to 1.0-1.2%")
        elif neutral_count > 70:
            logger.info(f"   ✅ Found {neutral_count} neutral days - MIGHT BE TOO LOOSE")
            logger.info(f"   This is {neutral_count/total*100:.1f}% of all days")
            logger.info(f"   Consider tightening slightly for better selectivity")
        else:
            logger.info(f"   ✅ Found {neutral_count} neutral days - GOOD BALANCE")
            logger.info(f"   This is {neutral_count/total*100:.1f}% of all days")
            logger.info(f"   Should generate 15-40 actual trades")
            logger.info(f"   Adequate sample size for statistical analysis")
        
        logger.info("=" * 80)
    
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
        Pattern identification with PRAGMATIC neutral structure filter.
        
        V4.2 focuses on getting adequate sample size while still being selective.
        """
        df['time_only'] = df.index.time
        t_open = pd.Timestamp("08:30").time()
        t_q1_end = pd.Timestamp("10:00").time()
        t_q2_start = pd.Timestamp(Q2_START_TIME).time()
        t_q2_end = pd.Timestamp(Q2_END_TIME).time()
        
        signals = []
        grouped = df.groupby(df.index.date)
        
        logger.info(f"\n{'='*80}")
        logger.info(f"🎯 V4.2: PRAGMATIC NEUTRAL STRUCTURE APPROACH")
        logger.info(f"{'='*80}")
        
        if USE_MOMENTUM_ONLY_FILTER:
            logger.info(f"Filter Mode: MOMENTUM ONLY (Simplified)")
            logger.info(f"   Momentum Threshold: <{MOMENTUM_ONLY_THRESHOLD*100:.1f}%")
        else:
            logger.info(f"Filter Mode: RANGE + MOMENTUM (Default)")
            logger.info(f"   Range Position: {NEUTRAL_RANGE_MIN:.2f}-{NEUTRAL_RANGE_MAX:.2f} (middle {int((NEUTRAL_RANGE_MAX-NEUTRAL_RANGE_MIN)*100)}%)")
            logger.info(f"   Momentum Threshold: <{NEUTRAL_MOMENTUM_MAX*100:.1f}%")
        
        logger.info(f"Time Window: {Q2_START_TIME}-{Q2_END_TIME}")
        logger.info(f"Target: Looking for 20-45 neutral days → 15-35 trades")
        logger.info(f"{'='*80}\n")
        
        neutral_days_count = 0
        directional_days_count = 0
        
        for date, day_data in tqdm(grouped, desc="🔬 Scanning for Pragmatic Neutral Days"):
            if len(day_data) < 30:
                continue
            
            if self.check_daily_loss_limit(pd.Timestamp(date)):
                continue
            
            daily_trades = len([s for s in signals if s['time'].date() == date])
            if daily_trades >= MAX_DAILY_TRADES:
                continue
            
            # Q1 Analysis with PRAGMATIC criteria
            q1_data = day_data[
                (day_data['time_only'] >= t_open) & 
                (day_data['time_only'] < t_q1_end)
            ]
            
            if len(q1_data) < MIN_Q1_BARS:
                continue
            
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
                    'range_vs_atr': q1_metrics['range_vs_atr']
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
                
                # Wick Sweep Long: Price sweeps BELOW Q1 low but closes ABOVE it
                if curr['low'] < q1_low and curr['close'] > q1_low:
                    signal = 1
                    setup_type = "Wick Sweep Long"
                    stop_loss = curr['low'] - (current_atr * ATR_STOP_MULTIPLIER)
                    
                    confidence = 1.0 if curr['volume'] >= q1_avg_volume * MIN_VOLUME_MULTIPLIER else 0.85
                    
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
        
        # Print comprehensive diagnostics
        self.print_comprehensive_diagnostics()
        
        logger.info(f"\n✅ Scan Complete:")
        logger.info(f"   Neutral Days Found: {neutral_days_count}")
        logger.info(f"   Directional Days Skipped: {directional_days_count}")
        logger.info(f"   Signals Found: {len(signals)}")
        
        if neutral_days_count > 0:
            signals_per_neutral_day = len(signals) / neutral_days_count
            logger.info(f"   Signals per Neutral Day: {signals_per_neutral_day:.2f}")
        
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
        """Execute trade with V4.2 parameters."""
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
            logger.info(f"⚠️  Only {len(self.trade_history)} trades - need 15+ for Monte Carlo")
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
        logger.info("=" * 80)
        logger.info("🚀 MOKSHA V4.2 - THE PRAGMATIC APPROACH")
        logger.info("=" * 80)
        logger.info("Translating Learning Into Profit")
        logger.info("=" * 80)
        
        df = self.fetch_real_data()
        if df.empty:
            return None
        
        signals = self.identify_wick_sweep_long_patterns(df)
        
        if signals.empty:
            logger.error("❌ No signals found even with relaxed criteria")
            logger.info("\n💡 The test period appears extremely trending/volatile.")
            logger.info("   Consider:")
            logger.info("      1. Testing on different time period (2022-2023)")
            logger.info("      2. Using MOMENTUM_ONLY mode")
            logger.info("      3. Removing structure filter entirely")
            return None
        
        logger.info(f"✅ Found {len(signals)} signals")
        
        equity = CAPITAL
        equity_curve = [CAPITAL]
        
        logger.info("\n" + "=" * 80)
        logger.info("💼 EXECUTING TRADES")
        logger.info("=" * 80)
        
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
        """Print comprehensive results."""
        logger.info("\n" + "=" * 80)
        logger.info("📊 V4.2 FINAL RESULTS")
        logger.info("=" * 80)
        
        logger.info(f"\n💰 CAPITAL PERFORMANCE")
        logger.info(f"   Starting Capital:     ${performance['capital']['starting']:,.2f}")
        logger.info(f"   Ending Capital:       ${performance['capital']['ending']:,.2f}")
        logger.info(f"   Total Return:         {performance['capital']['total_return']*100:.2f}%")
        logger.info(f"   Net Profit/Loss:      ${performance['capital']['total_pnl']:,.2f}")
        
        # Contextualize the result
        if performance['capital']['total_pnl'] > 0:
            logger.info(f"\n   ✅ PROFITABLE STRATEGY!")
            annual_return = performance['capital']['total_return']
            logger.info(f"   Annualized Return: ~{annual_return*100:.1f}% (based on ~1 year of data)")
        elif performance['capital']['total_pnl'] > -500:
            logger.info(f"\n   ⚠️  Near breakeven - marginal performance")
        else:
            logger.info(f"\n   ❌ Loss exceeded $500 - strategy needs work")
        
        logger.info(f"\n📈 TRADE STATISTICS")
        logger.info(f"   Total Trades:         {performance['trades']['total']}")
        logger.info(f"   Winners:              {performance['trades']['winners']}")
        logger.info(f"   Losers:               {performance['trades']['losers']}")
        logger.info(f"   Breakeven:            {performance['trades']['breakeven']}")
        logger.info(f"   Win Rate:             {performance['performance']['win_rate']*100:.1f}%")
        logger.info(f"   Profit Factor:        {performance['performance']['profit_factor']:.2f}")
        logger.info(f"   Expectancy per Trade: ${performance['performance']['expectancy']:.2f}")
        
        logger.info(f"\n💵 WIN/LOSS BREAKDOWN")
        logger.info(f"   Average Win:          ${performance['performance']['avg_win']:.2f}")
        logger.info(f"   Average Loss:         ${performance['performance']['avg_loss']:.2f}")
        logger.info(f"   Win/Loss Ratio:       {abs(performance['performance']['avg_win']/performance['performance']['avg_loss']):.2f}x" if performance['performance']['avg_loss'] != 0 else "N/A")
        logger.info(f"   Avg Transaction Cost: ${performance['risk']['avg_transaction_cost']:.2f}")
        
        logger.info(f"\n⚠️  RISK METRICS")
        logger.info(f"   Maximum Drawdown:     {performance['risk']['max_drawdown']*100:.2f}%")
        logger.info(f"   Sharpe Ratio:         {performance['risk']['sharpe_ratio']:.2f}")
        
        if mc_results:
            logger.info(f"\n🎲 MONTE CARLO SIMULATION ({MONTE_CARLO_RUNS:,} iterations)")
            logger.info(f"   Median Final Equity:  ${mc_results['final_equity']['median']:,.2f}")
            logger.info(f"   5th Percentile:       ${mc_results['final_equity']['percentile_5']:,.2f}")
            logger.info(f"   95th Percentile:      ${mc_results['final_equity']['percentile_95']:,.2f}")
            logger.info(f"   Expected Max DD:      {mc_results['max_drawdown']['median']*100:.2f}%")
            
            # Probability of profit
            prob_profit = sum(1 for e in range(len(mc_results['final_equity'])) if mc_results['final_equity'] > CAPITAL) / MONTE_CARLO_RUNS
            logger.info(f"   Probability of Profit: {prob_profit*100:.1f}%")
        
        if mae_mfe:
            logger.info(f"\n📉 MAE/MFE ANALYSIS")
            logger.info(f"   Winners:")
            logger.info(f"      Avg MAE (against): ${mae_mfe['winners']['avg_mae']:.2f}")
            logger.info(f"      Avg MFE (favor):   ${mae_mfe['winners']['avg_mfe']:.2f}")
            logger.info(f"   Losers:")
            logger.info(f"      Avg MAE (against): ${mae_mfe['losers']['avg_mae']:.2f}")
            logger.info(f"      Avg MFE (favor):   ${mae_mfe['losers']['avg_mfe']:.2f}")
        
        logger.info("\n" + "=" * 80)
        
        # Final verdict
        if performance['capital']['total_pnl'] > 1000 and performance['performance']['win_rate'] > 0.50:
            logger.info("✅ VERDICT: Strategy shows genuine edge")
            logger.info("   Next steps: Out-of-sample validation, then paper trading")
        elif performance['capital']['total_pnl'] > 0:
            logger.info("⚠️  VERDICT: Marginal edge detected")
            logger.info("   Next steps: Further optimization or combine with other strategies")
        else:
            logger.info("❌ VERDICT: Strategy lacks profitable edge in test period")
            logger.info("   Next steps: Test different time periods or new approaches")
        
        logger.info("=" * 80)


if __name__ == "__main__":
    backtest = MokshaV42Pragmatic()
    results = backtest.run()
    
    if results:
        output_file = f"/mnt/user-data/outputs/moksha_v42_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
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
                'mode': 'MOMENTUM_ONLY' if USE_MOMENTUM_ONLY_FILTER else 'RANGE+MOMENTUM',
                'neutral_range': f'{NEUTRAL_RANGE_MIN}-{NEUTRAL_RANGE_MAX}',
                'neutral_momentum': f'<{NEUTRAL_MOMENTUM_MAX*100}%',
                'time_window': f'{Q2_START_TIME}-{Q2_END_TIME}',
                'profit_target': f'{TAKE_PROFIT_RATIO}R'
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"\n💾 Results saved: {output_file}")
        logger.info("\n🎯 V4.2 BACKTEST COMPLETE - Check results above for edge validation")