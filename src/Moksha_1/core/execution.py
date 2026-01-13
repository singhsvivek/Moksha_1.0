# src/Moksha_1/core/execution.py
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, GetOrdersRequest
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus, OrderStatus # <--- CHANGED THIS
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestTradeRequest
from typing import Dict
import pandas as pd
import time
import math
from Moksha_1.config import settings

class ExecutionHandler:
    def __init__(self):
        self.client = TradingClient(
            settings.ALPACA_API_KEY,
            settings.ALPACA_SECRET_KEY,
            paper=True 
        )
        self.data_client = StockHistoricalDataClient(
            settings.ALPACA_API_KEY,
            settings.ALPACA_SECRET_KEY
        )
        print("🔌 Execution Handler: Connected to Alpaca.")

    def get_position_details(self, symbol: str) -> Dict:
        """
        Returns precise details: qty, available_qty, held_for_orders
        """
        try:
            pos = self.client.get_open_position(symbol)
            return {
                "qty": float(pos.qty),
                "available": float(pos.qty_available),
                "held": float(pos.qty) - float(pos.qty_available)
            }
        except Exception:
            # Position doesn't exist
            return {"qty": 0.0, "available": 0.0, "held": 0.0}

    def cancel_and_wait(self, symbol: str):
        """
        Cancels open orders and WAITS until 'held' shares are released.
        """
        try:
            # 1. Get Open Orders using Correct Enum
            req = GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[symbol])
            orders = self.client.get_orders(filter=req)
            
            if not orders:
                return

            print(f"   🧹 Clearing {len(orders)} blocking orders for {symbol}...")
            self.client.cancel_orders() # Cancels ALL open orders (simplest for clean slate)
            
            # 2. WAIT for 'Held' to drop to 0
            # Alpaca is fast, but not instant. We poll for up to 5 seconds.
            for _ in range(10):
                details = self.get_position_details(symbol)
                if details["held"] == 0:
                    break
                time.sleep(0.5)
            
            # Final Check
            details = self.get_position_details(symbol)
            if details["held"] > 0:
                print(f"   ⚠️ Warning: {details['held']} shares still held after cancel. Order might fail.")
            
        except Exception as e:
            print(f"   ⚠️ Cleanup Error: {e}")

    def execute_rebalance(self, decisions: pd.DataFrame, max_allocation: float = 0.20, live_run: bool = False):
        account = self.client.get_account()
        equity = float(account.equity)
        
        print(f"\n💰 Account Equity: ${equity:,.2f}")
        print("\n--- ⚡ EXECUTING ORDERS ---")
        
        for _, row in decisions.iterrows():
            symbol = row['symbol']
            signal = row['final_signal']
            target_value = equity * signal * max_allocation
            
            if live_run:
                self._place_atomic_order(symbol, target_value)
            else:
                self._place_order_dry_run(symbol, target_value)

    def _wait_for_fill(self, order_id: str, timeout: int = 10):
        """
        Polls the order status until it is FILLED or timeout.
        """
        print(f"      ⏳ Waiting for Leg 1 (ID: {order_id}) to fill...")
        for _ in range(timeout * 2): # Check every 0.5s
            order = self.client.get_order_by_id(order_id)
            if order.status == OrderStatus.FILLED:
                print("      ✅ Leg 1 Filled.")
                return True
            if order.status in [OrderStatus.CANCELED, OrderStatus.EXPIRED, OrderStatus.REJECTED]:
                print(f"      ❌ Leg 1 Failed ({order.status}). Stopping flip.")
                return False
            time.sleep(0.5)
        
        print("      ⚠️ Leg 1 timed out (not filled yet). Skipping Leg 2 to prevent conflict.")
        return False

    def _place_atomic_order(self, symbol: str, target_value: float):
        try:
            # Step A: Clean Slate
            self.cancel_and_wait(symbol)

            # Step B: Fresh Position State
            pos_details = self.get_position_details(symbol)
            current_qty = pos_details["qty"] 

            # Step C: Get Price & Target Shares
            trade_req = StockLatestTradeRequest(symbol_or_symbols=symbol)
            latest_trade = self.data_client.get_stock_latest_trade(trade_req)
            price = latest_trade[symbol].price
            
            print(f"👉 {symbol} (Held: {current_qty}): Signal -> Target ${target_value:,.2f}")

            # Calculate Target Shares
            target_shares_exact = target_value / price
            
            # Decide "Intended" Target (Integer for Shorts, Float for Longs)
            if target_shares_exact < 0:
                target_shares = round(target_shares_exact) 
            else:
                target_shares = round(target_shares_exact, 2)

            diff = target_shares - current_qty
            
            if abs(diff) < 0.01:
                print("   ✅ Position matches target.")
                return

            # --- HYBRID FLIP LOGIC (With Confirmation) ---
            is_flip_long_to_short = (current_qty > 0) and (target_shares < 0)
            is_flip_short_to_long = (current_qty < 0) and (target_shares > 0)

            if is_flip_long_to_short:
                print(f"   🔄 FLIP DETECTED (Long {current_qty} -> Short {target_shares})")
                
                # 1. Flatten Long
                if abs(current_qty) > 0:
                    print(f"      1. Flattening Long: Sell {current_qty}")
                    o1 = self.client.submit_order(MarketOrderRequest(
                        symbol=symbol, qty=abs(current_qty), side=OrderSide.SELL, time_in_force=TimeInForce.DAY
                    ))
                    # WAIT FOR FILL before Shorting
                    if not self._wait_for_fill(o1.id): return
                
                # 2. Open Short
                short_qty = abs(target_shares)
                if short_qty > 0:
                    print(f"      2. Opening Short: Sell {short_qty}")
                    self.client.submit_order(MarketOrderRequest(
                        symbol=symbol, qty=short_qty, side=OrderSide.SELL, time_in_force=TimeInForce.DAY
                    ))
                return

            if is_flip_short_to_long:
                print(f"   🔄 FLIP DETECTED (Short {current_qty} -> Long {target_shares})")
                
                # 1. Flatten Short
                if abs(current_qty) > 0:
                    print(f"      1. Flattening Short: Buy {abs(current_qty)}")
                    o1 = self.client.submit_order(MarketOrderRequest(
                        symbol=symbol, qty=abs(current_qty), side=OrderSide.BUY, time_in_force=TimeInForce.DAY
                    ))
                    # WAIT FOR FILL before Opening Long
                    if not self._wait_for_fill(o1.id): return
                
                # 2. Open Long
                if target_shares > 0:
                    print(f"      2. Opening Long: Buy {target_shares}")
                    self.client.submit_order(MarketOrderRequest(
                        symbol=symbol, qty=target_shares, side=OrderSide.BUY, time_in_force=TimeInForce.DAY
                    ))
                return

            # --- STANDARD TRADE ---
            side = OrderSide.BUY if diff > 0 else OrderSide.SELL
            qty_to_trade = abs(diff)
            
            if side == OrderSide.SELL and (current_qty < 0 or target_shares < 0):
                qty_to_trade = round(qty_to_trade)

            if qty_to_trade * price < 1.0: return

            print(f"   🚀 Sending Standard {side}: {qty_to_trade} shares")
            self.client.submit_order(MarketOrderRequest(
                symbol=symbol, qty=qty_to_trade, side=side, time_in_force=TimeInForce.DAY
            ))
            print(f"      ✅ Filled via Alpaca!")

        except Exception as e:
            print(f"      ❌ Execution Error: {e}")
            
    def _place_order_dry_run(self, symbol: str, target_value: float):
        action = "BUY" if target_value > 0 else "SELL"
        print(f"   [DRY RUN] Would {action} {symbol} target value ${abs(target_value):,.2f}")