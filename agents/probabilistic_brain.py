import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle
import os


class ProbabilisticBrain:
    def __init__(self, alpha=0.0005):
        self.alpha = alpha
        # 9 Inputs: [RSI, RSI_Slope, MACD, MACD_Slope, BB, BB_Slope, OBV, OBV_Slope, Bias]
        # 3 Actions: [0: BUY, 1: SELL, 2: HOLD]
        self.weights = np.zeros((9, 3))

        # Give HOLD a slight initial preference to prevent early over-trading
        self.weights[8, 2] = 0.0

        self.weight_history = []
        self.path = "outcomes/prob_weights.pkl"
        self.load()

    def get_probs(self, state):
        # Linear combination of state and weights
        scores = np.dot(state, self.weights)

        # TEMPERATURE SCALING:
        # 0.7 is a good 'conviction' level. Lower = more decisive.
        temperature = 0.7

        # Numerical stability: shift scores by max
        shifted_scores = (scores - np.max(scores)) / temperature
        exp_scores = np.exp(shifted_scores)
        return exp_scores / (np.sum(exp_scores) + 1e-9)

    def learn(self, state, action, discounted_reward):
        """
        Policy Gradient update.
        discounted_reward (G): The profit of the trade adjusted by Gamma and Pace.
        """
        probs = self.get_probs(state)

        # Compute Gradient of Log Softmax
        # d_softmax = (Probabilities - One-Hot-Action)
        d_softmax = probs.copy()
        d_softmax[action] -= 1

        # Weight Update: w = w - alpha * G * gradient
        # Note: If reward is positive, this increases the weight for the taken action.
        for i in range(len(state)):
            self.weights[i] -= self.alpha * discounted_reward * d_softmax * state[i]

        # Tracking for diagnostic plots
        # We only save history occasionally to save memory if updates are frequent
        # if len(self.weight_history) % 10 == 0:
        self.weight_history.append(self.weights.copy())

    def save(self):
        os.makedirs("outcomes", exist_ok=True)
        data = {"weights": self.weights, "history": self.weight_history}
        with open(self.path, "wb") as f:
            pickle.dump(data, f)

    def load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "rb") as f:
                    data = pickle.load(f)
                    if isinstance(data, dict):
                        self.weights = data["weights"]
                        self.weight_history = data.get("history", [])
                    else:
                        self.weights = data
            except Exception as e:
                print(f"Error loading weights: {e}. Starting fresh.")

    def evaluate_strategy(self, df_eval, test_pace=1):
        close = df_eval['Close'].values
        # Step through data
        indices = range(20, len(df_eval), test_pace)

        total_reward = 0.0
        in_position = False
        entry_price = 0.0
        trades_taken = 0

        # Match the agent's threshold
        def threshold(val, limit=0.01):
            return val if abs(val) > limit else 0.0

        for i in indices:
            # 1. Slice window (15 bars)
            window = df_eval.iloc[i - 15: i + 1]

            # 2. Extract and Normalize (Pace = 1 for Eval)
            def get_features(vec):
                # We use pace=1 for the evaluator to see 'raw' signal strength
                slope = (vec[-1] - np.mean(vec)) * np.sqrt(1)
                return vec[-1], slope

            r_val, r_slope = get_features(window['RSI_Scaled'].values)
            m_val, m_slope = get_features(window['MACD_Scaled'].values)
            b_val, b_slope = get_features(window['BB_Scaled'].values)
            o_val, o_slope = get_features(window['OBV_Scaled'].values)

            state = np.array([
                r_val, threshold(r_slope),
                m_val, threshold(m_slope),
                b_val, threshold(b_slope),
                o_val, threshold(o_slope),
                1.0  # Bias
            ], dtype=np.float64)

            probs = self.get_probs(state)

            # 3. CONVICTION LOGIC
            # Instead of argmax, we check if Buy/Sell can beat Hold + a small margin
            # This forces the brain to only act when it's 'sure'
            buy_score = probs[0]
            sell_score = probs[1]
            hold_score = probs[2]

            if buy_score > hold_score and not in_position:
                action = 0
            elif sell_score > hold_score and in_position:
                action = 1
            else:
                action = 2

            # 4. Execution
            if action == 0:
                in_position = True
                entry_price = close[i]
                trades_taken += 1
            elif action == 1:
                # Using simple percentage return for the quality score
                total_reward += (close[i] - entry_price) / entry_price
                in_position = False

        return total_reward if trades_taken > 0 else -0.01

    # --- VISUALIZATION TOOLS ---

    def plot_weights(self):
        # Updated to match the 9 features in MCAgent
        feature_names = [
            'RSI', 'RSI_Slope', 'MACD', 'MACD_Slope',
            'BB%', 'BB_Slope', 'OBV', 'OBV_Slope', 'Bias'
        ]
        action_names = ['BUY', 'SELL', 'HOLD']

        plt.figure(figsize=(12, 7))
        sns.heatmap(self.weights, annot=True, fmt=".3f", cmap="RdYlGn", center=0,
                    xticklabels=action_names, yticklabels=feature_names)
        plt.title("Monte Carlo Logic Map (Standardized Weights)")
        plt.show()

    def plot_decision_regions(self):
        """Visualizes how RSI and MACD interact to create decision zones."""
        x_rsi = np.linspace(-1.0, 1.0, 100)
        y_macd = np.linspace(-1.0, 1.0, 100)
        xx, yy = np.meshgrid(x_rsi, y_macd)
        grid_actions = np.zeros(xx.shape)

        for i in range(xx.shape[0]):
            for j in range(xx.shape[1]):
                # Dummy state for other features
                state = np.array([xx[i, j], 0, yy[i, j], 0, 0, 0, 0, 0, 1.0])
                grid_actions[i, j] = np.argmax(self.get_probs(state))

        plt.figure(figsize=(8, 6))
        # cmap matches standard: 0=Red(Buy), 1=Green(Sell), 2=Yellow(Hold)
        # Note: Depending on your choice index, you might need to adjust cmap
        plt.pcolormesh(xx, yy, grid_actions, cmap='brg', alpha=0.3, shading='auto')

        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='blue', alpha=0.5, label='BUY'),
            Patch(facecolor='red', alpha=0.5, label='SELL'),
            Patch(facecolor='green', alpha=0.5, label='HOLD')
        ]
        plt.legend(handles=legend_elements)
        plt.xlabel("RSI (Scaled)")
        plt.ylabel("MACD (Scaled)")
        plt.title("Universal Decision Zones (Monte Carlo Context)")
        plt.show()

    def plot_momentum_logic(self):
        x_val = np.linspace(-1.0, 1.0, 100)
        y_slope = np.linspace(-0.5, 0.5, 100)  # Widened slope range
        xx, yy = np.meshgrid(x_val, y_slope)
        grid = np.zeros(xx.shape)

        for i in range(xx.shape[0]):
            for j in range(xx.shape[1]):
                # State for RSI and its Slope
                state = np.array([xx[i, j], yy[i, j], 0, 0, 0, 0, 0, 0, 1.0])
                probs = self.get_probs(state)

                # Use argmax here just to see the 'raw' winner
                grid[i, j] = np.argmax(probs)

        plt.figure(figsize=(10, 7))
        # Corrected Colormap: RdYlGn (Red=Sell, Yellow=Hold, Green=Buy)
        # Note: Check if 0=Buy, 1=Sell, 2=Hold matches your index!
        plt.pcolormesh(xx, yy, grid, cmap='RdYlGn', alpha=0.6, shading='auto')
        plt.xlabel("Indicator Value")
        plt.ylabel("Momentum (Slope)")
        plt.title("Brain Logic Map: Value vs. Momentum")
        plt.colorbar(label="0: BUY, 1: SELL, 2: HOLD")
        plt.show()