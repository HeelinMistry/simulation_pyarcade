# 🧠 Unified World Model: Strategic Architecture

This project implements a **MuZero-inspired World Model** for financial time-series trading, optimized for high-conviction decision making and rigorous drawdown avoidance.

## 🎯 Strategic Goals
- **Structural Awareness**: Beyond simple momentum, the model analyzes market volatility (ATR) and mean-reversion signals (Mean Deviation).
- **Extreme Conviction**: Uses Entropy Regularization and Hyper-Sharpening (Temp 0.05) to eliminate indecisive "mushy" trading.
- **Risk Defense**: Implements a "Gradient Cliff" to penalize major drawdowns (>2%) with 5x severity.
- **Selective Perception**: Employs Hard Pruning and L1 Regularization to ignore market noise and focus on high-alpha patterns.

---

## 🏗️ Core Components

### 1. `UnifiedWorldModel` (The Brain)
A tri-network architecture accelerated via CuPy, now featuring a **92-feature input vector**.
*   **Representation ($h$):** Compresses 6 indicators across 5 time-paces (90 market features) + 2 portfolio features into a 128-dim latent state.
*   **Dynamics ($g$):** The "Hallucination Engine." Predicts the next market state and expected reward based on a specific action signal.
*   **Prediction ($f$):** Outputs the high-conviction **Policy ($\vec{p}$)** and the **Value ($v$)** (expected total future reward).

### 2. `MCTSPlanner` (The Strategist)
Performs deep mental rollouts to "look before it leaps."
*   **Deep Rollouts:** Simulates 50-100 steps into the future for every potential action.
*   **Gamma (0.99):** Prioritizes long-term trend sustainability over immediate price wiggles.
*   **Softmax Temperature (0.05-0.15):** Transforms subtle value differences into near-categorical training targets (winner-take-all).

### 3. `UnifiedExecutor` (The Operator)
The execution bridge with live-trading safety filters.
*   **Live Conviction Filter:** Requires >40% probability before initiating new positions to prevent jitter.
*   **Live Temperature (0.20):** Introduces deliberate "strategic doubt" during live execution to prevent regime-lock (e.g., short-only bias).

---

## 🚦 Feature Engineering (The 92-Feature Vector)
The model processes 6 primary indicators across 5 time horizons (Paces 1, 2, 4, 8, 12):
1. **RSI**: Overbought/Oversold momentum.
2. **MACD**: Trend convergence/divergence.
3. **Bollinger**: Volatility-relative price position.
4. **OBV Velocity**: Net volume flow rate (13-bar velocity).
5. **ATR Volatility**: Price range expansion detector.
6. **Mean Deviation**: Distance from the 20-period SMA (Mean Reversion signal).

---

## 📈 Reward Shaping & Risk Management
The training loop uses an amplified signal (x400) to force value differentiation:
- **Standard Reward**: `PNL * 400`.
- **Commitment Penalty**: `-0.1` fixed penalty for trades lasting < 15 ticks.
- **Drawdown Penalty**: 
    - `PNL < 0`: 2.0x multiplier.
    - `PNL < -0.02`: 5.0x multiplier (Hyper-penalty for major losses).

---

## 🛠️ Diagnostics & Health Markers
Monitor these markers in `diagnostic.py` to ensure model health:
- **Action Separation (> 1.5)**: Measures if the brain sees LONG and SHORT as different futures.
- **Predicted Entropy (< 1.5 bits)**: Lower values indicate higher trading conviction.
- **Repr Sparsity (10% - 20%)**: High sparsity indicates the model has successfully pruned noise.
- **Latent Drift (~0.9x)**: Ensures hallucinations remain stable over deep rollouts.
