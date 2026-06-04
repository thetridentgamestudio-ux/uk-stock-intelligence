"""
Feature pruning recommendations (SHAP/importance-based).

Currently: all 55 features are being used.

Next steps for ~+0.5 to +0.8pp accuracy improvement:
1. Identify highly correlated features and remove one of each pair
2. Remove features with <0.1% individual contribution to predictions
3. Combine related features (e.g., multiple moving averages into single "MA trend" feature)

For now: keep all 55 features since backtest shows 53.8% with stacking meta-learner.

Real improvement will come from:
  - Monthly retraining (captures regime shifts)
  - Regime-conditional models (BULL vs BEAR vs NEUTRAL)
  - More high-quality signal sources (insider buying, short interest)
"""
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Feature pruning deferred until after 4+ weeks of live accuracy data.")
logger.info("Current model: 55 features + stacking meta-learner = 53.8% backtest accuracy")
logger.info("Priority: Monthly retraining + regime-conditional models for real improvement")
