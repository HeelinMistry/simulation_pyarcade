from agents.UnifiedWorldModel import UnifiedWorldModel
import numpy as np
import matplotlib.pyplot as plt
import cupy as cp
import os


def analyze_world_model(paces=(1, 2, 4, 8, 12)):
    # UPDATED: 6 indicators * 3 states * 5 paces + 2 portfolio = 92
    num_indicators = 6
    input_size = (num_indicators * 3 * len(paces)) + 2
    
    model = UnifiedWorldModel(input_size=input_size)

    if not os.path.exists("outcomes/best_world_model.pkl"):
        print("❌ No World Model file found.")
        return
    model.load("outcomes/best_world_model.pkl")

    w_repr = cp.asnumpy(model.W_repr)
    w_dyn = cp.asnumpy(model.W_dyn)
    w_pred = cp.asnumpy(model.W_pred)

    fig = plt.figure(figsize=(24, 16))
    gs = fig.add_gridspec(2, 3)

    # --- 1. Representation Heatmap ---
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(w_repr, aspect='auto', cmap='magma')
    ax1.set_title("Repr Sensitivity (Clutter Check)")
    fig.colorbar(im1, ax=ax1)

    # --- 2. Dynamics Engine ---
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(w_dyn, aspect='auto', cmap='bwr')
    ax2.set_title("Dynamics (Physics Stability)")
    fig.colorbar(im2, ax=ax2)

    # --- 3. Prediction Mapping ---
    ax3 = fig.add_subplot(gs[0, 2])
    im3 = ax3.imshow(w_pred.T, aspect='auto', cmap='RdYlGn')
    ax3.set_title("Prediction (Pi/V Mapping)")
    ax3.set_yticks([0, 1, 2, 3, 4])
    ax3.set_yticklabels(["LONG", "SHORT", "CLOSE", "HOLD", "VALUE"])
    fig.colorbar(im3, ax=ax3)

    # --- 4. Hallucination Test ---
    ax4 = fig.add_subplot(gs[1, :2])
    test_input = cp.random.randn(1, input_size).astype(cp.float32)
    current_latent = model.get_initial_state(test_input)
    rewards_path = []
    for _ in range(50):
        current_latent, rew = model.simulate_next(current_latent, 0.0)
        rewards_path.append(float(cp.asnumpy(rew).item()))
    ax4.plot(rewards_path, color='purple', label="Hallucinated P/L")
    ax4.axhline(0, color='black', linestyle='--')
    ax4.set_title("Rollout Drift Check")
    ax4.legend()

    # --- 5. Global Weight Distributions ---
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.hist(w_repr.flatten(), bins=100, alpha=0.5, label="Repr", color='blue')
    ax5.hist(w_dyn.flatten(), bins=100, alpha=0.5, label="Dyn", color='red')
    ax5.set_title("Weight Distribution (Sparsity Check)")
    ax5.legend()

    plt.tight_layout()
    plt.savefig("outcomes/world_model_diagnostic.png")
    
    run_numerical_health_check(model, w_repr, w_dyn, w_pred, input_size)
    plt.show()


def run_numerical_health_check(model, w_repr, w_dyn, w_pred, input_size):
    print("\n--- 🩺 WORLD MODEL HEALTH & BIAS CHECK ---")
    
    # 1. Sparsity
    get_sparse = lambda w: np.sum(np.abs(w) < 1e-5) / w.size
    print(f"Repr Sparsity: {get_sparse(w_repr):.1%} | Dyn: {get_sparse(w_dyn):.1%} | Pred: {get_sparse(w_pred):.1%}")

    # 2. Hallucination Drift
    test_input = cp.random.randn(10, input_size).astype(cp.float32)
    z = model.get_initial_state(test_input)
    mags = [float(cp.linalg.norm(z))]
    for _ in range(20):
        z, _ = model.simulate_next(z, 0.0)
        mags.append(float(cp.linalg.norm(z)))
    print(f"Latent Drift (20-steps): {mags[-1]/(mags[0]+1e-9):.2f}x")

    # 3. DIRECTIONAL BIAS
    test_batch = cp.random.randn(500, input_size).astype(cp.float32)
    probs, _ = model.predict(model.get_initial_state(test_batch))
    actions = cp.argmax(probs, axis=1)
    
    long_count = int(cp.sum(actions == 0))
    short_count = int(cp.sum(actions == 1))
    neutral_count = int(cp.sum((actions == 2) | (actions == 3)))
    
    print(f"\n--- ⚖️ DIRECTIONAL BIAS (over 500 random inputs) ---")
    print(f"LONG Preference:  {long_count/500:.1%}")
    print(f"SHORT Preference: {short_count/500:.1%}")
    print(f"NEUTRAL (C/H):   {neutral_count/500:.1%}")
    
    # 4. INDICATOR SENSITIVITY (Updated for 6 indicators)
    sensitivities = np.sum(np.abs(w_repr), axis=1)
    # Each indicator has 3 states (raw, slope, std) repeated across 5 paces.
    # Total input: 6 indicators * 3 states * 5 paces = 90 + 2 portfolio.
    
    print(f"\n--- 🔍 FEATURE SENSITIVITY (Avg Weight Magnitude) ---")
    indicator_names = ["RSI", "MACD", "Bollinger", "OBV Velocity", "ATR Volatility", "Mean Dev"]
    for i, name in enumerate(indicator_names):
        # Calculate indices for this indicator across all paces
        indices = []
        for pace in range(5):
            # pace_start = pace * 18
            # indicator_start = pace_start + i
            # We want raw(0), slope(6), and std(12) for this indicator
            base = pace * 18 + i
            indices.extend([base, base + 6, base + 12])
        
        avg_sens = np.mean(sensitivities[indices])
        print(f"{name:<15}: {avg_sens:.4f}")

    port_sens = np.mean(sensitivities[-2:])
    print(f"{'PORTFOLIO':<15}: {port_sens:.4f}")
    print("---------------------------------------------\n")

if __name__ == "__main__":
    analyze_world_model()
