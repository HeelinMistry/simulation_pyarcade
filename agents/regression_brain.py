import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle
import os


class RegressionBrain:
    def __init__(self, alpha=0.0001):  # Regression usually needs a lower alpha
        self.alpha = alpha
        # 9 Inputs: [RSI_S, RSI_V, MACD_S, MACD_V, BB_S, BB_V, OBV_S, OBV_V, Bias]
        # 1 Output: Expected Unrealized Profit (Continuous)
        self.weights = np.zeros(9)

        self.weight_history = []
        self.path = "outcomes/regression_weights.pkl"
        self.load()

    def predict(self, state):
        """Returns the predicted unrealized profit for a given state."""
        return np.dot(state, self.weights)

    def learn(self, state, actual_profit):
        """
        Updates weights using Mean Squared Error (MSE) gradient descent.
        actual_profit: The unrealized profit reported by an agent.
        """
        prediction = self.predict(state)

        # Error = (Actual - Predicted)
        error = actual_profit - prediction

        # Gradient = -2 * error * state (we simplify the constant 2 into alpha)
        update = self.alpha * error * state

        # Clipping to maintain stability against price spikes
        self.weights += np.clip(update, -0.01, 0.01)

        # Track history (storing mean absolute weight to track 'Growth')
        self.weight_history.append(self.weights.copy())

    def save(self):
        os.makedirs("outcomes", exist_ok=True)
        data = {"weights": self.weights, "history": self.weight_history}
        with open(self.path, "wb") as f:
            pickle.dump(data, f)

    def load(self):
        if os.path.exists(self.path):
            with open(self.path, "rb") as f:
                data = pickle.load(f)
                if isinstance(data, dict):
                    self.weights = data["weights"]
                    self.weight_history = data.get("history", [])
                else:
                    self.weights = data

    def plot_logic_map(self):
        feature_names = ['RSI_S', 'RSI_V', 'MACD_S', 'MACD_V', 'BB_S', 'BB_V', 'OBV_S', 'OBV_V', 'Bias']

        plt.figure(figsize=(6, 8))
        # We reshape to (9, 1) for a single-column heatmap
        sns.heatmap(self.weights.reshape(-1, 1), annot=True, fmt=".4f",
                    cmap="RdYlGn", center=0, yticklabels=feature_names)
        plt.title("Indicator Influence on Expected Profit")
        plt.show()

    def plot_profit_sensitivity(self):
        """Visualizes how RSI affects predicted profit across its range."""
        rsi_range = np.linspace(-1.0, 1.0, 100)
        predictions = []

        for r in rsi_range:
            # Create a neutral state except for RSI
            state = np.array([r, 0, 0, 0, 0, 0, 0, 0, 1.0])
            predictions.append(self.predict(state))

        plt.figure(figsize=(8, 4))
        plt.plot(rsi_range, predictions, color='blue', lw=2)
        plt.axhline(0, color='black', ls='--')
        plt.fill_between(rsi_range, predictions, 0, where=(np.array(predictions) > 0), color='green', alpha=0.3)
        plt.fill_between(rsi_range, predictions, 0, where=(np.array(predictions) < 0), color='red', alpha=0.3)
        plt.xlabel("RSI Smooth (Input)")
        plt.ylabel("Predicted Unrealized Profit")
        plt.title("Profit Expectancy Curve")
        plt.show()