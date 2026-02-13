import unittest
from unittest.mock import MagicMock
import pandas as pd
# Import your strategies (ensure paths are correct in your dev env)
from src.production_equity import KineticScalarProduction
from src.production_silver import SilverStructureProduction

class TestMokshaLive(unittest.TestCase):
    def test_equity_entry(self):
        """Tests if Equity Bot buys when Z-Score < -2.0"""
        bot = KineticScalarProduction()
        bot.api = MagicMock()
        
        # Mock Data: Ratio at 2.0, Mean 2.2, Std 0.05 -> Z ~ -4.0 (Buy)
        data = {'z_score': -3.0, 'ratio': 2.0, 'trend_sma': 1.9}
        row = pd.Series(data)
        
        # Mock Account
        bot.api.list_positions.return_value = []
        bot.api.get_account.return_value.equity = '25000'
        
        bot.reconcile_positions(row)
        
        # Should submit 2 buy orders (Tech + Hedge)
        self.assertEqual(bot.api.submit_order.call_count, 2)
        print("✅ Equity Test Passed: Buy Signal Triggered")

    def test_silver_reclaim(self):
        """Tests if Silver Bot buys on Reclaim + Volume"""
        bot = SilverStructureProduction()
        bot.api = MagicMock()
        
        # Setup Q1 Levels
        bot.q1_levels = {'low': 10.0, 'high': 20.0}
        
        # Mock Bar: Low(9.0) < Q1_Low(10.0), Close(10.5) > Q1_Low, High Volume
        mock_bars = pd.DataFrame([{
            'low': 9.0, 'high': 11.0, 'close': 10.5, 'volume': 1000
        }])
        # Mock API responses
        bot.api.get_bars.return_value.df = mock_bars
        # Mock ATR calculation
        bot.get_atr = MagicMock(return_value=0.5)
        
        bot.check_signals()
        
        # Should submit 1 bracket order
        bot.api.submit_order.assert_called_once()
        print("✅ Silver Test Passed: Reclaim Signal Triggered")

if __name__ == '__main__':
    unittest.main()