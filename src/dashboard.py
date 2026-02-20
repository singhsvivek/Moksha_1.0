import streamlit as st
import pandas as pd
import sys
import os
import time
import signal
import alpaca_trade_api as tradeapi
from datetime import datetime

# --- CONFIGURATION ---
STRATEGIES = {
    "equity": {
        "name": "Kinetic Scalar", 
        "file": "src/production_equity.py", 
        "pid": "/tmp/moksha_equity.pid", 
        "desc": "Mean Reversion: TQQQ/SQQQ/UPRO/SPXU"
    },
    "silver": {
        "name": "Silver Structure", 
        "file": "src/production_silver.py", 
        "pid": "/tmp/moksha_silver.pid", 
        "desc": "Liquidity Trap: AGQ"
    }
}

# --- PAGE SETUP ---
st.set_page_config(page_title="MOKSHA COMMAND", layout="wide", page_icon="🦁")

# --- URBAN LIGHT THEME CSS ---
st.markdown("""
<style>
    .stApp { background-color: #FAFAFA; color: #1F1F1F; font-family: sans-serif; }
    .urban-card {
        background: #FFFFFF; border: 1px solid #E5E5E5; border-radius: 12px;
        padding: 24px; box-shadow: 0 2px 8px rgba(0,0,0,0.03); margin-bottom: 20px;
    }
    .metric-label { font-size: 12px; font-weight: 700; color: #888; text-transform: uppercase; letter-spacing: 0.8px; }
    .metric-value { font-size: 36px; font-weight: 800; color: #111; line-height: 1; margin-top: 8px; }
    .metric-sub { font-size: 14px; font-weight: 500; margin-top: 8px; }
    .pos { color: #00C853; }
    .neg { color: #FF3D00; }
    .strat-badge-on { background: #E8F5E9; color: #2E7D32; font-size: 11px; font-weight: 700; padding: 4px 10px; border-radius: 20px; }
    .strat-badge-off { background: #F5F5F5; color: #757575; font-size: 11px; font-weight: 700; padding: 4px 10px; border-radius: 20px; }
    .log-container { 
        background: #FFFFFF; color: #333; font-family: monospace; font-size: 11px; 
        height: 450px; overflow-y: auto; padding: 15px; border-radius: 8px; border: 1px solid #E0E0E0;
    }
    .log-entry { border-bottom: 1px dashed #F0F0F0; padding-bottom: 4px; margin-bottom: 4px; display: flex; }
    .log-trade { color: #2962FF; font-weight: 700; background: #E3F2FD; padding: 2px 6px; border-radius: 4px; }
    .log-error { color: #D50000; font-weight: 700; background: #FFEBEE; padding: 2px 6px; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)

# --- SYSTEM UTILS ---
def get_pid_status(key):
    pid_file = STRATEGIES[key]['pid']
    if os.path.exists(pid_file):
        try:
            with open(pid_file, 'r') as f: pid = int(f.read().strip())
            os.kill(pid, 0)
            return True, pid
        except: return False, None
    return False, None

def toggle_bot(key, action):
    cfg = STRATEGIES[key]
    pid_file = cfg['pid']
    if action == "START":
        os.system("mkdir -p /app/logs")
        # FIXED: Removed '>> /app/logs/moksha.log 2>&1' to prevent double logging
        # The script's internal logger already writes to the file.
        cmd = f"nohup python -u {cfg['file']} > /dev/null 2>&1 & echo $! > {pid_file}"
        os.system(cmd)
        time.sleep(1)
    elif action == "STOP":
        running, pid = get_pid_status(key)
        if running and pid:
            try: os.kill(pid, signal.SIGKILL)
            except: pass
        if os.path.exists(pid_file): os.remove(pid_file)
    st.rerun()

def get_alpaca_data():
    """Connects DIRECTLY to Alpaca to bypass backend complexity."""
    try:
        api = tradeapi.REST(
            os.getenv('ALPACA_API_KEY'),
            os.getenv('ALPACA_SECRET_KEY'),
            os.getenv('ALPACA_BASE_URL'),
            api_version='v2'
        )
        return api.get_account(), api.list_positions()
    except Exception:
        return None, []

def parse_logs():
    lines = []
    log_file = "/app/logs/moksha.log"
    if not os.path.exists(log_file): return "Waiting for system logs..."
    try:
        with open(log_file, "r") as f: raw = f.readlines()[-100:]
        for line in reversed(raw):
            if not line.strip(): continue
            parts = line.split('|')
            ts = parts[0].strip()[:19] if len(parts) > 1 else datetime.now().strftime("%H:%M:%S")
            msg = parts[-1].strip() if len(parts) > 1 else line.strip()
            
            css = ""
            if "🚀" in line or "💰" in line: css = "log-trade"
            elif "❌" in line: css = "log-error"
            elif "👀" in line: css = "log-trade"
            
            lines.append(f'<div class="log-entry"><span style="color:#999; margin-right:10px; min-width:65px;">{ts}</span><span class="{css}">{msg}</span></div>')
    except: lines = ["Error reading logs."]
    return "".join(lines)

# --- UI LAYOUT ---

st.title("🦁 MOKSHA COMMAND")
st.markdown("---")

acct, positions = get_alpaca_data()

if acct:
    equity = float(acct.equity)
    pnl = equity - float(acct.last_equity)
    pnl_pct = (pnl / float(acct.last_equity)) * 100
    bp = float(acct.buying_power)
else:
    equity, bp, pnl, pnl_pct = 0.0, 0.0, 0.0, 0.0

# 1. Metrics
c1, c2, c3 = st.columns(3)
with c1:
    sign = "+" if pnl >= 0 else ""
    css = "pos" if pnl >= 0 else "neg"
    st.markdown(f"""<div class="urban-card"><div class="metric-label">Net Liquidity</div><div class="metric-value">${equity:,.2f}</div><div class="metric-sub {css}">{sign}${pnl:,.2f} ({sign}{pnl_pct:.2f}%)</div></div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""<div class="urban-card"><div class="metric-label">Buying Power</div><div class="metric-value">${bp:,.2f}</div><div class="metric-sub" style="color:#888;">4x Intraday Leverage</div></div>""", unsafe_allow_html=True)
with c3:
    status_color = "#00C853" if acct else "#FF3D00"
    status_text = "SYSTEM ONLINE" if acct else "DISCONNECTED"
    st.markdown(f"""<div class="urban-card"><div class="metric-label">System Status</div><div class="metric-value" style="color:{status_color}; font-size:28px;">{status_text}</div><div class="metric-sub" style="color:#888;">Alpaca API v2</div></div>""", unsafe_allow_html=True)

# 2. Main Content
col_control, col_data = st.columns([1, 2])

with col_control:
    st.markdown("### ⚙️ Engines")
    for key, cfg in STRATEGIES.items():
        running, pid = get_pid_status(key)
        status_html = '<span class="strat-badge-on">ACTIVE</span>' if running else '<span class="strat-badge-off">OFFLINE</span>'
        st.markdown(f"""<div class="urban-card" style="padding: 20px;"><div style="display:flex; justify-content:space-between; margin-bottom:10px;"><span style="font-weight:700;">{cfg['name']}</span>{status_html}</div><div style="font-size:13px; color:#666;">{cfg['desc']}</div></div>""", unsafe_allow_html=True)
        b1, b2 = st.columns(2)
        if b1.button("Start Engine", key=f"start_{key}", disabled=running, type="primary", use_container_width=True): toggle_bot(key, "START")
        if b2.button("Kill Engine", key=f"stop_{key}", disabled=not running, type="secondary", use_container_width=True): toggle_bot(key, "STOP")
        st.write("")

with col_data:
    st.markdown("### 📊 Holdings")
    if positions:
        data = []
        for p in positions:
            data.append({
                "Symbol": p.symbol, 
                "Qty": float(p.qty), 
                "Entry": float(p.avg_entry_price), 
                "Price": float(p.current_price), 
                "P&L": float(p.unrealized_pl), 
                "Return %": float(p.unrealized_plpc)*100
            })
        df = pd.DataFrame(data)
        st.dataframe(df.style.format({"Qty": "{:.2f}", "Entry":"${:.2f}", "Price":"${:.2f}", "P&L":"${:+.2f}", "Return %":"{:+.2f}%"}).applymap(lambda x: f"color: {'#00C853' if x > 0 else '#FF3D00'}; font-weight:700", subset=["P&L", "Return %"]), use_container_width=True, hide_index=True)
    else: st.info("Portfolio is 100% Cash.")
    
    st.write("")
    st.markdown("### 📜 System Logs")
    st.markdown(f'<div class="log-container">{parse_logs()}</div>', unsafe_allow_html=True)
    if st.button("Clear Logs"):
        os.system("truncate -s 0 /app/logs/moksha.log")
        st.rerun()

time.sleep(5)
st.rerun()