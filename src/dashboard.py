import streamlit as st
import pandas as pd
import sys
import os
import time
import signal

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from Moksha_1.core.execution import ExecutionHandler
except ImportError: pass

st.set_page_config(page_title="MOKSHA COMMAND", layout="wide", page_icon="🏦")

st.markdown("""
<style>
    .stApp { background-color: #f8fafc; color: #1e293b; font-family: 'Inter', sans-serif; }
    .vault-card { background: white; border: 1px solid #e2e8f0; border-radius: 8px; padding: 15px; box-shadow: 0 1px 2px rgba(0,0,0,0.05); }
    .metric-val { font-size: 1.5rem; font-weight: 700; color: #0f172a; }
    .metric-lbl { font-size: 0.75rem; color: #64748b; text-transform: uppercase; font-weight: 600; }
    .strat-card { background: white; border-left: 4px solid #cbd5e1; border-radius: 4px; padding: 15px; margin-bottom: 10px; border: 1px solid #e2e8f0; }
    .strat-active { border-left-color: #10b981; }
    .strat-inactive { border-left-color: #ef4444; }
    .log-box { background: #1e293b; color: #f8fafc; font-family: monospace; font-size: 0.8rem; height: 300px; overflow-y: auto; padding: 10px; border-radius: 6px; }
    .log-trade { color: #4ade80; font-weight: bold; }
    .log-warn { color: #facc15; }
    [data-testid="stDataFrame"] { border: 1px solid #e2e8f0; border-radius: 6px; }
</style>
""", unsafe_allow_html=True)

STRATEGIES = {
    "equity": {"name": "Equity Engine", "file": "src/etf_arb_loop.py", "pid": "/tmp/moksha_equity.pid", "desc": "TQQQ/SQQQ Arbitrage"},
    "midas": {"name": "Midas Protocol", "file": "src/commodity_arb_loop.py", "pid": "/tmp/moksha_midas.pid", "desc": "Gold/Silver Volatility"}
}

@st.cache_resource
def get_handlers():
    try: return ExecutionHandler()
    except: return None

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
    if action == "START":
        os.system("mkdir -p /app/logs")
        
        # --- THE FIX: DIRECT INJECTION ---
        # We use 'env' to set PYTHONPATH inline. No script file needed.
        # This forces the python process to look in /app/libs
        cmd = f"nohup env PYTHONPATH=/app/libs python -u {cfg['file']} >> /app/logs/moksha.log 2>&1 & echo $! > {cfg['pid']}"
        
        os.system(cmd)
        
    elif action == "STOP":
        run, pid = get_pid_status(key)
        if run and pid: 
            try: os.kill(pid, signal.SIGKILL)
            except: pass
        if os.path.exists(cfg['pid']): os.remove(cfg['pid'])
    time.sleep(1)

def parse_logs():
    lines = []
    try:
        with open("/app/logs/moksha.log", "r") as f:
            raw = f.readlines()[-60:]
        for line in reversed(raw):
            css = ""
            if "ENTRY" in line or "EXIT" in line: css = "log-trade"
            elif "WARNING" in line: css = "log-warn"
            if "Connected to Alpaca" in line: continue 
            lines.append(f'<div class="{css}">{line.strip()}</div>')
    except: lines = ["Waiting for logs..."]
    return "".join(lines)

st.title("🏦 MOKSHA COMMAND")
st.markdown("---")

handler = get_handlers()
if handler:
    try:
        acct = handler.client.get_account()
        equity, bp, pnl = float(acct.equity), float(acct.buying_power), float(acct.equity) - float(acct.last_equity)
    except: equity, bp, pnl = 0,0,0
else: equity, bp, pnl = 0,0,0

c1, c2, c3 = st.columns(3)
c1.markdown(f'<div class="vault-card"><div class="metric-lbl">Net Liquidity</div><div class="metric-val">${equity:,.2f}</div></div>', unsafe_allow_html=True)
c2.markdown(f'<div class="vault-card"><div class="metric-lbl">Buying Power</div><div class="metric-val">${bp:,.2f}</div></div>', unsafe_allow_html=True)
c3.markdown(f'<div class="vault-card"><div class="metric-lbl">Daily P&L</div><div class="metric-val" style="color:{"#10b981" if pnl >= 0 else "#ef4444"}">${pnl:,.2f}</div></div>', unsafe_allow_html=True)

st.markdown("---")
c_strat, c_log = st.columns([1, 1])

with c_strat:
    st.subheader("📡 Strategy Engines")
    for key, cfg in STRATEGIES.items():
        running, pid = get_pid_status(key)
        css = "strat-active" if running else "strat-inactive"
        status = f"ONLINE (PID {pid})" if running else "OFFLINE"
        
        st.markdown(f"""
        <div class="strat-card {css}">
            <div style="font-weight:bold; font-size:1.1rem;">{cfg['name']}</div>
            <div style="color:#64748b; font-size:0.8rem;">{cfg['desc']}</div>
            <div style="margin-top:5px; font-family:monospace; font-size:0.8rem;">{status}</div>
        </div>
        """, unsafe_allow_html=True)
        
        b1, b2 = st.columns(2)
        if st.button("▶ ENGAGE", key=f"start_{key}", disabled=running):
            toggle_bot(key, "START")
            st.rerun()
        if st.button("⏹ KILL", key=f"stop_{key}", disabled=not running):
            toggle_bot(key, "STOP")
            st.rerun()

with c_log:
    st.subheader("📜 Live Feed")
    st.markdown(f'<div class="log-box">{parse_logs()}</div>', unsafe_allow_html=True)
    if st.button("Clear Logs"):
        os.system("truncate -s 0 /app/logs/moksha.log")
        st.rerun()

st.markdown("---")
st.subheader("💼 Active Holdings")
if handler:
    try:
        positions = handler.get_current_positions()
        if not positions.empty:
            df = positions[['symbol', 'qty', 'current_price', 'unrealized_pl', 'unrealized_plpc']].copy()
            df.columns = ['SYMBOL', 'QTY', 'PRICE', 'P&L ($)', 'P&L (%)']
            def style_pnl(val): return f'color: {"#10b981" if val > 0 else "#ef4444"}; font-weight: bold;'
            st.dataframe(df.style.applymap(style_pnl, subset=['P&L ($)', 'P&L (%)']).format({'PRICE': '${:.2f}', 'P&L ($)': '${:+.2f}', 'P&L (%)': '{:+.2f}%'}), use_container_width=True, hide_index=True)
        else: st.info("Portfolio is 100% Cash.")
    except Exception as e: st.error(f"Error: {e}")

time.sleep(3)
st.rerun()
