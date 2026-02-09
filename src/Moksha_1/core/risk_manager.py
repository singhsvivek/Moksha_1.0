import pandas as pd
from Moksha_1.utils.logger import logger

class RiskManager:
    def __init__(self, client=None):
        self.client = client
        self.cppi_floor_pct = 0.80  # Protect 80% of capital
        self.cppi_multiplier = 3.0  # Aggressiveness

    def apply_cppi(self, target_allocation, current_equity, peak_equity):
        """
        Constant Proportion Portfolio Insurance (CPPI)
        Allocates to risky assets based on the 'Cushion' above the floor.
        """
        floor_value = peak_equity * self.cppi_floor_pct
        cushion = current_equity - floor_value
        
        if cushion < 0:
            logger.warning("🛡️ CPPI ALERT: Breached Floor! Moving to 100% Cash.")
            return 0.0
            
        risk_budget = cushion * self.cppi_multiplier
        
        # Cap allocation at 100% (or 1.0)
        allocation_pct = min(risk_budget / current_equity, 1.0)
        allocation_pct = max(allocation_pct, 0.0)
        
        return allocation_pct

    def validate_signals(self, decisions, current_equity):
        # Existing logic...
        # [Keep your existing validation logic here]
        return decisions