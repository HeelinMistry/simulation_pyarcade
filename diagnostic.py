from agents.UnifiedWorldModel import UnifiedWorldModel
import numpy as np
import matplotlib.pyplot as plt
import cupy as cp
import os


def analyze_world_model(paces=(1, 2, 4, 8, 12)):
    # 1. Setup Input Dimensions
    input_size = (len(paces) * 12) + 2
    model = UnifiedWorldModel(input_size=input_size)

    if not os.path.exists("outcomes/best_world_model.pkl"):
        print("❌ No World Model file found at outcomes/best_world_model.pkl")
        return
    model.load("outcomes/best_world_model.pkl")

    # Move weights to CPU
    w_repr = cp.asnumpy(model.W_repr)  # [Input -> Latent]
    w_dyn = cp.asnumpy(model.W_dyn)  # [(Latent + Action) -> (Latent + Reward)]
    w_pred = cp.asnumpy(model.W_pred)  # [Latent -> (Policy + Value)]

    fig = plt.figure(figsize=(24, 14))
    gs = fig.add_gridspec(2, 3)

    # --- 1. Representation Heatmap (How we see the market) ---
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(w_repr, aspect='auto', cmap='magma')
    ax1.set_title(f"Representation Sensitivity ({input_size} -> 128)")
    ax1.set_ylabel("Input Features")
    ax1.set_xlabel("Latent Neurons")
    # Add Pace Dividers
    for i in range(len(paces)):
        ax1.axhline(i * 12 - 0.5, color='white', alpha=0.3, linewidth=0.5)
    ax1.axhline(input_size - 2.5, color='cyan', alpha=0.8, linewidth=1.5)  # Portfolio line
    fig.colorbar(im1, ax=ax1)

    # --- 2. Dynamics Engine (The Physics of the Hallucination) ---
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(w_dyn, aspect='auto', cmap='bwr')  # Blue-White-Red for pos/neg
    ax2.set_title("Dynamics Weights (State+Action -> Next)")
    ax2.set_ylabel("Latent State + Action Signal")
    ax2.set_xlabel("Predicted Next State + Reward")
    # Highlight the 'Action' input row (the very last input row)
    ax2.axhline(w_dyn.shape[0] - 1.5, color='black', linewidth=2)
    # Highlight the 'Reward' output column (the very last column)
    ax2.axvline(w_dyn.shape[1] - 1.5, color='black', linewidth=2)
    fig.colorbar(im2, ax=ax2)

    # --- 3. Prediction Head (Policy & Value Mapping) ---
    ax3 = fig.add_subplot(gs[0, 2])
    im3 = ax3.imshow(w_pred.T, aspect='auto', cmap='RdYlGn')
    ax3.set_title("Prediction Mapping (Latent -> Pi/V)")
    ax3.set_yticks([0, 1, 2, 3, 4])
    ax3.set_yticklabels(["LONG", "SHORT", "CLOSE", "HOLD", "VALUE"])
    ax3.set_xlabel("Latent Neuron Index")
    fig.colorbar(im3, ax=ax3)

    # --- 4. Hallucination Test: Trajectory Consistency ---
    # We feed a random state and simulate 50 steps of 'HOLD' to see if it drifts to infinity
    ax4 = fig.add_subplot(gs[1, :2])
    test_input = cp.random.randn(1, input_size).astype(cp.float32)
    current_latent = model.get_initial_state(test_input)

    rewards_path = []
    for _ in range(50):
        # action_signal = 0.0 for 'HOLD'
        current_latent, rew = model.simulate_next(current_latent, 0.0)

        # Use .item() or [0,0] to extract the scalar value from the CuPy array
        val = float(cp.asnumpy(rew).item())
        rewards_path.append(val)

    ax4.plot(rewards_path, color='purple', linewidth=2, label="Hallucinated Reward (HOLD)")# --- 5. Weight Health (Sparsity check) ---
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.hist(w_repr.flatten(), bins=50, alpha=0.5, label="Repr", color='blue')
    ax5.hist(w_dyn.flatten(), bins=50, alpha=0.5, label="Dyn", color='red')
    ax5.hist(w_pred.flatten(), bins=50, alpha=0.5, label="Pred", color='green')
    ax5.set_title("Global Weight Distributions")
    ax5.legend()

    plt.tight_layout()
    plt.savefig("outcomes/world_model_diagnostic.png")
    plt.show()


if __name__ == "__main__":
    analyze_world_model()