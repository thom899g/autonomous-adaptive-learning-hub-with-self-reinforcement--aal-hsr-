"""
Configuration management for AAL-HSR trading system.
Centralizes all configurable parameters with validation and type safety.
"""
import os
from dataclasses import dataclass
from typing import Optional, Dict, List
from enum import Enum
import logging

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


class TradingAction(Enum):
    """Defined trading actions for RL agent"""
    HOLD = 0
    BUY = 1
    SELL = 2
    SCALE_IN = 3
    SCALE_OUT = 4


@dataclass
class ModelConfig:
    """Reinforcement learning model configuration"""
    algorithm: str = "PPO"
    policy: str = "MlpPolicy"
    learning_rate: float = 0.0003
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    entropy_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    device: str = "auto"
    
    def validate(self) -> bool:
        """Validate model configuration parameters"""
        valid_algorithms = ["PPO", "A2C", "DQN"]
        if self.algorithm not in valid_algorithms:
            logger.error(f"Invalid algorithm: {self.algorithm}. Must be one of {valid_algorithms}")
            return False
        if not 0 < self.learning_rate <= 1:
            logger.error(f"Invalid learning_rate: {self.learning_rate}. Must be between 0 and 1")
            return False
        if self.gamma <= 0 or self.gamma >= 1:
            logger.error(f"Invalid gamma: {self.gamma}. Must be between 0 and 1")
            return False
        return True


@dataclass
class TradingConfig:
    """Trading-specific configuration"""
    initial_balance: float = 10000.0
    max_position_size: float = 0.25  # 25% of portfolio
    transaction_fee: float = 0.001  # 0.1% per transaction
    slippage: float = 0.0005  # 0.05% slippage
    risk_free_rate: float = 0.02  # Annual risk-free rate
    max_drawdown_limit: float = 0.20  # Stop trading at 20% drawdown
    trade_size_strategy: str = "percentage"  # or "fixed"
    
    # Reward calculation weights
    reward_weights: Dict[str, float] = None
    
    def __post_init__(self):
        if self.reward_weights is None:
            self.reward_weights = {
                "profit": 0.4,