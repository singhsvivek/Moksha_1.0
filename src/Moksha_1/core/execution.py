from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, GetOrdersRequest
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus, OrderStatus
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestTradeRequest
from typing import Dict, List, Tuple, Optional
import pandas as pd
import time
from datetime import datetime, timezone
from Moksha_1.config import settings
from Moksha_1.utils.logger import logger
from Moksha_1.utils.messenger import messenger

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
        logger.info("🔌 Execution Handler: Connected to Alpaca.")

    # --- 1. SYSTEM HEALTH & DASHBOARD SUPPORT ---
    def check_market_status(self) -> bool:
        try:
            clock = self.client.get_clock()
            if clock.is_open:
                return True
            
            # If closed, log details
            next_open = clock.next_open.replace(tzinfo=timezone.utc).astimezone()
            time_to_open = next_open - datetime.now().astimezone()
            logger.warning(f"⛔ Market is CLOSED. Next Open: {next_open} (in {time_to_open})")
            return False
        except Exception as e:
            logger.error(f"Failed to fetch market clock: {e}")
            return False 

    def get_current_positions(self) -> pd.DataFrame:
        """
        NEW: Fetches current open positions for the Dashboard Visualization.
        """
        try:
            positions = self.client.get_all_positions()
            if not positions:
                return pd.DataFrame()
            
            data = []
            for p in positions:
                data.append({
                    'symbol': p.symbol,
                    'qty': float(p.qty),
                    'avg_entry_price': float(p.avg_entry_price),
                    'current_price': float(p.current_price),
                    'unrealized_pl': float(p.unrealized_pl),
                    'unrealized_plpc': float(p.unrealized_plpc) * 100
                })
            return pd.DataFrame(data)
        except Exception as e:
            logger.error(f"❌ Failed to fetch positions for dashboard: {e}")
            return pd.DataFrame()

    # --- 2. EXECUTION HELPERS (RESTORED) ---
    def get_position_details(self, symbol: str) -> Dict:
        """Returns precise details: qty, available_qty, held_for_orders"""
        try:
            pos = self.client.get_open_position(symbol)
            return {
                "qty": float(pos.qty),
                "available": float(pos.qty_available),
                "held": float(pos.qty) - float(pos.qty_available)
            }
        except Exception:
            return {"qty": 0.0, "available": 0.0, "held": 0.0}

    def get_all_positions(self) -> Dict[str, float]:
        try:
            positions = self.client.get_all_positions()
            return {p.symbol: float(p.qty) for p in positions}
        except Exception as e:
            logger.error(f"Failed to fetch positions: {e}")
            return {}

    def get_latest_prices(self, symbols: List[str]) -> Dict[str, float]:
        try:
            req = StockLatestTradeRequest(symbol_or_symbols=symbols)
            trades = self.data_client.get_stock_latest_trade(req)
            return {sym: trade.price for sym, trade in trades.items()}
        except Exception as e:
            logger.error(f"Failed to fetch prices: {e}")
            return {}

    def cancel_and_wait(self, symbol: str):
        """Cancels open orders and WAITS until 'held' shares are released."""
        try:
            req = GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[symbol])
            orders = self.client.get_orders(filter=req)
            if not orders: return

            logger.info(f"   🧹 Clearing {len(orders)} blocking orders for {symbol}...")
            self.client.cancel_orders()
            
            for _ in range(10):
                details = self.get_position_details(symbol)
                if details["held"] == 0: break
                time.sleep(0.5)
        except Exception as e:
            logger.error(f"   ⚠️ Cleanup Error: {e}")

    def _wait_for_fill(self, order_id: str, timeout: int = 10) -> bool:
        logger.info(f"      ⏳ Waiting for Leg 1 (ID: {order_id}) to fill...")
        for _ in range(timeout * 2):
            try:
                order = self.client.get_order_by_id(order_id)
                if order.status == OrderStatus.FILLED:
                    logger.info("      ✅ Leg 1 Filled.")
                    return True
                if order.status in [OrderStatus.CANCELED, OrderStatus.EXPIRED, OrderStatus.REJECTED]:
                    logger.error(f"      ❌ Leg 1 Failed ({order.status}). Stopping flip.")
                    return False
                time.sleep(0.5)
            except Exception:
                time.sleep(0.5)
        logger.warning("      ⚠️ Leg 1 timed out. Skipping Leg 2.")
        return False

    def _wait_for_buying_power(self, target_increase: float, timeout: int = 20):
        if target_increase < 100: return
        logger.info(f"      ⏳ Verifying Cash Settlement (Expected +${target_increase:,.2f})...")
        try:
            start_bp = float(self.client.get_account().buying_power)
            for i in range(timeout):
                current_bp = float(self.client.get_account().buying_power)
                if current_bp > start_bp + (target_increase * 0.5):
                    logger.info(f"      ✅ Funds Settled. New BP: ${current_bp:,.2f}")
                    return
                time.sleep(1)
            logger.warning(f"      ⚠️ BP update timed out. Proceeding...")
        except Exception as e:
            logger.error(f"      ⚠️ Failed to poll buying power: {e}")

    # --- 3. MAIN EXECUTION LOGIC (RESTORED) ---
    def execute_rebalance(self, decisions: pd.DataFrame, max_allocation: float = 0.20, live_run: bool = False):
        executed_trades = []
        start_equity = 0.0
        
        try:
            # SAFETY CHECK
            if live_run and not self.check_market_status():
                messenger.send_message("Attempted to trade while market is CLOSED.", title="⛔ Trade Aborted")
                return

            # FETCH ACCOUNT
            logger.info("   ⏳ Connecting to Alpaca to fetch Account Equity...")
            account = self.client.get_account()
            start_equity = float(account.equity)
            buying_power = float(account.buying_power)
            
            logger.info(f"   ✅ Account Retrieved. Equity: ${start_equity:,.2f} | BP: ${buying_power:,.2f}")
            logger.info("   --- ⚡ STARTING ORDER EXECUTION CYCLE ---")
            
            # PRE-PROCESS & SORT
            all_symbols = decisions['symbol'].tolist()
            current_positions = self.get_all_positions()
            current_prices = self.get_latest_prices(all_symbols)
            
            trade_plan = []
            
            for _, row in decisions.iterrows():
                symbol = row['symbol']
                signal = row['final_signal']
                target_value = start_equity * signal * max_allocation
                
                qty = current_positions.get(symbol, 0.0)
                price = current_prices.get(symbol, 0.0)
                
                if price == 0: 
                    trade_plan.append((symbol, target_value, 0))
                    continue
                    
                current_value = qty * price
                estimated_change = target_value - current_value
                trade_plan.append((symbol, target_value, estimated_change))

            # SORT: Sells first
            trade_plan.sort(key=lambda x: x[2]) 
            
            # EXECUTE LOOP
            for symbol, target_value, estimated_change in trade_plan:
                direction = "SELL/SHORT" if estimated_change < 0 else "BUY/COVER"
                logger.info(f"   🔎 Processing {symbol}: Est. Change ${estimated_change:,.2f} ({direction})")
                
                if live_run:
                    trade_result = self._place_atomic_order(symbol, target_value)
                    if trade_result:
                        executed_trades.append(trade_result)
                    
                    # Wait for funds if significant sell
                    if estimated_change < -500:
                        self._wait_for_buying_power(abs(estimated_change))
                else:
                    self._place_order_dry_run(symbol, target_value)
            
            # REPORTING
            end_equity = float(self.client.get_account().equity)
            messenger.send_execution_report(executed_trades, start_equity, end_equity)
            logger.info("   --- ✅ EXECUTION CYCLE COMPLETE ---")

        except Exception as e:
            logger.error(f"Execution Cycle Failed: {e}", exc_info=True)
            messenger.send_message(f"Critical Execution Failure: {e}", title="❌ Execution Error")

    def _submit_market_order(self, symbol: str, qty: float, side: OrderSide, price_est: float):
        try:
            order_data = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=side,
                time_in_force=TimeInForce.DAY
            )
            order = self.client.submit_order(order_data)
            logger.info(f"      ✅ Submitted {side} {qty} {symbol}")
            return order
        except Exception as e:
            logger.error(f"      ❌ API Error: {e}")
            return None

    def _place_atomic_order(self, symbol: str, target_value: float) -> Optional[Dict]:
        """Handles the complex trade logic (Flips, exact share calc)."""
        trade_summary = None
        try:
            self.cancel_and_wait(symbol)
            pos_details = self.get_position_details(symbol)
            current_qty = pos_details["qty"] 

            # Get Fresh Price
            trade_req = StockLatestTradeRequest(symbol_or_symbols=symbol)
            latest_trade = self.data_client.get_stock_latest_trade(trade_req)
            price = latest_trade[symbol].price
            
            logger.info(f"👉 {symbol} (Held: {current_qty}): Signal -> Target ${target_value:,.2f}")

            # Target Shares Calc
            target_shares_exact = target_value / price
            # Integer constraint for shorts
            target_shares = round(target_shares_exact) if target_shares_exact < 0 else round(target_shares_exact, 2)

            diff = target_shares - current_qty
            if abs(diff) < 0.01:
                logger.info("   ✅ Position matches target.")
                return None

            # FLIP LOGIC
            is_flip_long_to_short = (current_qty > 0) and (target_shares < 0)
            is_flip_short_to_long = (current_qty < 0) and (target_shares > 0)

            if is_flip_long_to_short:
                logger.info(f"   🔄 FLIP DETECTED (Long {current_qty} -> Short {target_shares})")
                if abs(current_qty) > 0:
                    o1 = self._submit_market_order(symbol, abs(current_qty), OrderSide.SELL, price)
                    if not o1 or not self._wait_for_fill(o1.id): return None
                
                short_qty = abs(target_shares)
                if short_qty > 0:
                    o2 = self._submit_market_order(symbol, short_qty, OrderSide.SELL, price)
                    if o2: trade_summary = {"symbol": symbol, "side": "SELL (Flip)", "qty": short_qty, "price": price}
                return trade_summary

            if is_flip_short_to_long:
                logger.info(f"   🔄 FLIP DETECTED (Short {current_qty} -> Long {target_shares})")
                if abs(current_qty) > 0:
                    o1 = self._submit_market_order(symbol, abs(current_qty), OrderSide.BUY, price)
                    if not o1 or not self._wait_for_fill(o1.id): return None
                
                if target_shares > 0:
                    o2 = self._submit_market_order(symbol, target_shares, OrderSide.BUY, price)
                    if o2: trade_summary = {"symbol": symbol, "side": "BUY (Flip)", "qty": target_shares, "price": price}
                return trade_summary

            # STANDARD TRADE
            side = OrderSide.BUY if diff > 0 else OrderSide.SELL
            qty_to_trade = abs(diff)
            if side == OrderSide.SELL and (current_qty < 0 or target_shares < 0):
                qty_to_trade = round(qty_to_trade)

            if qty_to_trade * price < 1.0: return None

            logger.info(f"   🚀 Sending Standard {side}: {qty_to_trade} shares")
            o = self._submit_market_order(symbol, qty_to_trade, side, price)
            
            if o:
                trade_summary = {"symbol": symbol, "side": str(side), "qty": qty_to_trade, "price": price}
            
            return trade_summary

        except Exception as e:
            logger.error(f"      ❌ Execution Error: {e}", exc_info=True)
            return None

    def _place_order_dry_run(self, symbol: str, target_value: float):
        action = "BUY" if target_value > 0 else "SELL"
        logger.info(f"   [DRY RUN] Would {action} {symbol} target value ${abs(target_value):,.2f}")