from agents.unified_brain import UnifiedBrain
import numpy as np
import matplotlib.pyplot as plt
import cupy as cp
import os

def analyze_brain_v3(paces=(1, 2, 3, 4, 5)):
    # 1. Update Input Size: (5 paces * 12 features) + 2 portfolio context (Position, UPNL)
    # This aligns with your 63-feature requirement.
    input_size = (len(paces) * 12) + 2

    # Note: Ensure your UnifiedBrain class is updated to handle output_size=4 internally
    brain = UnifiedBrain(input_size=input_size)
    if not os.path.exists("outcomes/best_unified_brain.pkl"):
        print("❌ No brain file found.")
        return
    brain.load()

    # Move weights to CPU for plotting
    w1 = cp.asnumpy(brain.W1)
    w2 = cp.asnumpy(brain.W2)

    fig = plt.figure(figsize=(22, 12))
    gs = fig.add_gridspec(2, 3)

    # --- 1. Weight Distribution ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(w1.flatten(), bins=50, color='skyblue', alpha=0.7, label="W1 (Input)")
    ax1.hist(w2.flatten(), bins=50, color='orange', alpha=0.7, label="W2 (Output)")
    ax1.set_title("Neural Health (Weight Spread)")
    ax1.legend()

    # --- 2. Recalibrated Feature Heatmap (63 Inputs) ---
    ax2 = fig.add_subplot(gs[0, 1:])
    im = ax2.imshow(w1, aspect='auto', cmap='magma')
    ax2.set_title(f"Feature Sensitivity ({input_size} Inputs)")

    sub_features = ["cRSI", "cMACD", "cBB", "cOBV", "sRSI", "sMACD", "sBB", "sOBV", "vRSI", "vMACD", "vBB", "vOBV"]
    yticks = []
    yticklabels = []

    # Label Market Paces (0-59)
    for i, p in enumerate(paces):
        base = i * 12
        yticks.append(base + 5.5)
        yticklabels.append(f"Pace {p}")
        ax2.axhline(base - 0.5, color='white', linewidth=1.0, alpha=0.6)

    # Label Portfolio Context (60-62)
    # Stronger divider at index 60
    ax2.axhline(60 - 0.5, color='cyan', linewidth=1.5, alpha=0.8)
    yticks.append(61)
    yticklabels.append("PORTFOLIO")

    ax2.set_yticks(yticks)
    ax2.set_yticklabels(yticklabels, fontsize=10, fontweight='bold')

    # Right side labels for specific features
    ax2_right = ax2.twinx()
    ax2_right.set_ylim(ax2.get_ylim())
    ax2_right.set_yticks(np.arange(input_size))
    # Full list: 60 market features + 2 portfolio
    right_labels = (sub_features * len(paces)) + ["POSITION", "UPNL"]
    ax2_right.set_yticklabels(right_labels, fontsize=7, alpha=0.6)

    fig.colorbar(im, ax=ax2)

    # --- 3. Confidence Histogram ---
    dummy_input = cp.random.randn(1000, input_size).astype(cp.float32)
    probs, _ = brain.forward(dummy_input)
    p_np = cp.asnumpy(probs)
    confidences = np.max(p_np, axis=1)

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.hist(confidences, bins=30, color='purple', alpha=0.6)
    # Update random baseline to 0.25 (1/4 actions)
    ax3.axvline(0.25, color='red', linestyle='--', label="Random (0.25)")
    ax3.set_title("Brain Conviction (Max Prob)")
    ax3.legend()

    # --- 4. Global Action Tendency (Pie Chart) ---
    ax4 = fig.add_subplot(gs[1, 1])
    avg_p = p_np.mean(axis=0)
    # Updated labels and colors for 4 actions
    action_labels = ["LONG", "SHORT", "CLOSE", "HOLD"]
    action_colors = ['#2ecc71', '#e74c3c', '#f39c12', '#95a5a6']
    ax4.pie(avg_p, labels=action_labels, colors=action_colors, autopct='%1.1f%%')
    ax4.set_title("Global Action Tendency")

    # --- 5. Hidden Layer -> Action Mapping (W2) ---
    ax5 = fig.add_subplot(gs[1, 2])
    # w2 is [Hidden_Size, 4], so w2.T is [4, Hidden_Size]
    ax5.imshow(w2.T, aspect='auto', cmap='RdYlGn')
    ax5.set_yticks([0, 1, 2, 3])
    ax5.set_yticklabels(action_labels)
    ax5.set_title("Hidden Logic (W2 Mapping)")
    ax5.set_xlabel("Hidden Neuron Index")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    analyze_brain_v3()