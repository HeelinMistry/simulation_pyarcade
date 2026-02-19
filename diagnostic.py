import numpy as np
import matplotlib.pyplot as plt
import cupy as cp
import os
from agents.unified_brain import UnifiedBrain


def analyze_brain_v2(paces=(1, 2, 4, 8, 16)):
    input_size = len(paces) * 8
    brain = UnifiedBrain(input_size=input_size)
    if not os.path.exists("outcomes/unified_brain.pkl"):
        print("❌ No brain file found.")
        return
    brain.load()

    # Move weights to CPU
    w1 = cp.asnumpy(brain.W1)
    w2 = cp.asnumpy(brain.W2)

    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(2, 3)

    # --- 1. Weight Distribution (Stayed the same, it was good) ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(w1.flatten(), bins=50, color='skyblue', alpha=0.7, label="W1")
    ax1.hist(w2.flatten(), bins=50, color='orange', alpha=0.7, label="W2")
    ax1.set_title("Neural Health (Weight Spread)")
    ax1.legend()

    # --- 2. Labeled Feature Heatmap (The Improvement) ---
    ax2 = fig.add_subplot(gs[0, 1:])
    im = ax2.imshow(w1, aspect='auto', cmap='magma')
    ax2.set_title("Feature Sensitivity (Rows = Inputs, Cols = Hidden Neurons)")

    # Create labels for the Y-axis
    feature_names = ["RSI", "MACD", "BB", "OBV", "sRSI", "sMACD", "sBB", "sOBV"]
    yticks = []
    yticklabels = []
    for i, p in enumerate(paces):
        yticks.append(i * 8 + 4)  # Put label in the middle of the block
        yticklabels.append(f"Pace {p}")

    ax2.set_yticks(yticks)
    ax2.set_yticklabels(yticklabels, fontsize=10, fontweight='bold')
    # Add horizontal lines to separate paces visually
    for i in range(1, len(paces)):
        ax2.axhline(i * 8 - 0.5, color='white', linewidth=0.5, alpha=0.5)

    fig.colorbar(im, ax=ax2)

    # --- 3. Confidence Histogram (New!) ---
    # Test with 1000 random states to see conviction levels
    dummy_input = cp.random.randn(1000, input_size)
    probs, _ = brain.forward(dummy_input)
    p_np = cp.asnumpy(probs)
    confidences = np.max(p_np, axis=1)

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.hist(confidences, bins=30, color='purple', alpha=0.6)
    ax3.axvline(0.33, color='red', linestyle='--', label="Random Guess")
    ax3.set_title("Brain Conviction (Max Prob)")
    ax3.set_xlabel("Confidence Level")
    ax3.legend()

    # --- 4. Action Specificity (Improved Bias Plot) ---
    ax4 = fig.add_subplot(gs[1, 1])
    avg_p = p_np.mean(axis=0)
    ax4.pie(avg_p, labels=["BUY", "SELL", "HOLD"], colors=['#2ecc71', '#e74c3c', '#95a5a6'], autopct='%1.1f%%')
    ax4.set_title("Global Action Tendency")

    # --- 5. Output Strength (New!) ---
    # Shows which hidden neurons drive which actions
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.imshow(w2.T, aspect='auto', cmap='RdYlGn')
    ax5.set_yticks([0, 1, 2])
    ax5.set_yticklabels(["BUY", "SELL", "HOLD"])
    ax5.set_title("Hidden Layer -> Action Mapping")
    ax5.set_xlabel("Hidden Neuron Index")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    analyze_brain_v2()