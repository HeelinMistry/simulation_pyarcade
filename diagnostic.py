import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cupy as cp
import pandas as pd

from agents.UnifiedWorldModel import UnifiedWorldModel
from agents.MCTSPlanner import MCTSPlanner

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
MASTER_CSV   = "data/processed/XRPUSDT_master_processed.csv"
MODEL_PATH   = "outcomes/best_world_model.pkl"
FEATURES     = ["RSI_Scaled", "MACD_Scaled", "BB_Scaled",
                "OBV_Scaled", "ATR_Scaled", "MeanDev_Scaled"]
PACES        = (1, 2, 4, 8, 12)
NUM_IND      = len(FEATURES)               # 6
INPUT_SIZE   = (NUM_IND * 3 * len(PACES)) + 2  # 92
SAMPLE_N     = 500                         # rows used for distribution checks
ACTION_NAMES = ["LONG", "SHORT", "CLOSE", "HOLD"]
MIN_CONVICTION = 0.45                      # must match unified_executor


# ─────────────────────────────────────────────
# 1. Load model + data
# ─────────────────────────────────────────────
def load_model():
    model = UnifiedWorldModel(input_size=INPUT_SIZE)
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"❌ No model found at {MODEL_PATH}")
    model.load(MODEL_PATH)
    print(f"✅ Model loaded from {MODEL_PATH}")
    return model


def load_real_data():
    """Returns (indicators_np [N,92_raw_features], prices_np [N])
    NOTE: indicators here are the raw 6-feature rows, NOT the aggregated
    92-d state.  We build real 92-d states inside the checks using the
    StateAggregator so results match exactly what training sees."""
    if not os.path.exists(MASTER_CSV):
        raise FileNotFoundError(f"❌ Master CSV not found at {MODEL_PATH}")
    df = pd.read_csv(MASTER_CSV)
    ind = df[FEATURES].values.astype(np.float32)
    prices = df["Close"].values.astype(np.float32)
    print(f"✅ Real data loaded: {len(ind):,} rows")
    return ind, prices


def build_real_states(indicators, prices, n=SAMPLE_N):
    """
    Build n real 92-d state vectors using the same StateAggregator
    pipeline the executor uses, so distributions are authentic.
    Samples evenly spaced across the validation half of the data.
    """
    from agents.state_aggregator import StateAggregator
    from agents.unified_executor import UnifiedExecutor
    from agents.MCTSPlanner import MCTSPlanner

    # Dummy planner/executor just to drive aggregator
    model_tmp = UnifiedWorldModel(input_size=INPUT_SIZE)
    planner_tmp = MCTSPlanner(model_tmp, lookahead_depth=1)
    executor = UnifiedExecutor("diag", planner_tmp, paces=PACES)

    split = int(len(indicators) * 0.8)
    val_ind = indicators[split:]
    val_p   = prices[split:]

    # warm up from a safe starting point
    warm_idx = 200
    executor.aggregator.warm_up_all(val_ind, warm_idx)

    step_size = max(1, (len(val_ind) - warm_idx) // n)
    states, sample_prices, sample_returns = [], [], []

    for i in range(n):
        idx = warm_idx + i * step_size
        if idx + 1 >= len(val_ind):
            break
        executor.current_side = None
        executor.inventory    = []
        state = executor.get_state(val_ind[idx], val_p[idx])
        states.append(state)
        sample_prices.append(val_p[idx])
        if idx + 1 < len(val_p):
            ret = (val_p[idx + 1] - val_p[idx]) / val_p[idx]
            sample_returns.append(ret)

    return (np.array(states,         dtype=np.float32),
            np.array(sample_prices,  dtype=np.float32),
            np.array(sample_returns, dtype=np.float32))


# ─────────────────────────────────────────────
# 2. Visual diagnostic panels (existing + extended)
# ─────────────────────────────────────────────
def plot_visual_diagnostics(model, real_states_np):
    w_repr1    = cp.asnumpy(model.W_repr1)
    w_repr2    = cp.asnumpy(model.W_repr2)
    w_repr3    = cp.asnumpy(model.W_repr3)
    w_dyn_state  = cp.asnumpy(model.W_dyn_state)
    w_dyn_reward = cp.asnumpy(model.W_dyn_reward)
    w_dyn      = np.concatenate([w_dyn_state, w_dyn_reward], axis=1)
    w_pred     = cp.asnumpy(model.W_pred)

    # Calculate end-to-end sensitivity for the representation network
    # This approximates the Jacobian by chaining absolute weight matrices
    end_to_end_repr_sensitivity = np.abs(w_repr1) @ np.abs(w_repr2) @ np.abs(w_repr3)

    fig = plt.figure(figsize=(28, 20))
    gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.42, wspace=0.35)

    # --- Panel 1: Repr heatmap (end-to-end sensitivity) ---
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(end_to_end_repr_sensitivity, aspect='auto', cmap='magma')
    ax1.set_title("Repr End-to-End Sensitivity")
    ax1.set_xlabel("Final latent unit")
    ax1.set_ylabel("Input feature")
    fig.colorbar(im1, ax=ax1, shrink=0.8)

    # --- Panel 2: Dynamics heatmap ---
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(w_dyn, aspect='auto', cmap='bwr')
    ax2.set_title("Dynamics (State + Reward heads)")
    ax2.set_xlabel("Output")
    ax2.set_ylabel("Hidden + action")
    fig.colorbar(im2, ax=ax2, shrink=0.8)

    # --- Panel 3: Prediction mapping ---
    ax3 = fig.add_subplot(gs[0, 2])
    im3 = ax3.imshow(w_pred.T, aspect='auto', cmap='RdYlGn')
    ax3.set_title("Prediction Head (Pi / V)")
    ax3.set_yticks([0, 1, 2, 3, 4])
    ax3.set_yticklabels(ACTION_NAMES + ["VALUE"])
    fig.colorbar(im3, ax=ax3, shrink=0.8)

    # --- Panel 4: Weight distributions ---
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.hist(w_repr1.flatten(),     bins=120, alpha=0.5, label="Repr1",     color='darkblue')
    ax4.hist(w_repr2.flatten(),     bins=120, alpha=0.5, label="Repr2",     color='royalblue')
    ax4.hist(w_repr3.flatten(),     bins=120, alpha=0.5, label="Repr3",     color='lightsteelblue')
    ax4.hist(w_dyn_state.flatten(), bins=120, alpha=0.5, label="Dyn_State", color='tomato')
    ax4.hist(w_dyn_reward.flatten(),bins=120, alpha=0.5, label="Dyn_Reward",color='gold')
    ax4.set_title("Weight Distributions (Sparsity)")
    ax4.set_xlabel("Weight value")
    ax4.legend(fontsize=8)

    # --- Panel 5: Rollout drift ---
    ax5 = fig.add_subplot(gs[1, :2])
    test_z = model.get_initial_state(
        cp.random.randn(5, INPUT_SIZE).astype(cp.float32))
    rewards_path = []
    norms_path   = []
    for _ in range(60):
        test_z, rew = model.simulate_next(test_z, 0.0)
        rewards_path.append(float(cp.asnumpy(rew).mean()))
        norms_path.append(float(cp.linalg.norm(test_z)))
    ax5.plot(rewards_path, color='purple',  label="Mean hallucinated reward")
    ax5.plot(norms_path,   color='teal',    label="Latent norm", linestyle='--')
    ax5.axhline(0, color='black', linestyle=':')
    ax5.set_title("Rollout Drift (60 steps) — should stay near 0")
    ax5.legend(fontsize=8)

    # --- Panel 6: Entropy distribution on REAL states ---
    ax6 = fig.add_subplot(gs[1, 2])
    real_batch  = cp.asarray(real_states_np, dtype=cp.float32)
    latents     = model.get_initial_state(real_batch)
    probs_cp, _ = model.predict(latents)
    probs_np    = cp.asnumpy(probs_cp)
    entropy     = -np.sum(probs_np * np.log2(probs_np + 1e-9), axis=1)
    ax6.hist(entropy, bins=40, color='darkorange', edgecolor='black')
    ax6.axvline(entropy.mean(),  color='red',   linestyle='--', label=f"Mean {entropy.mean():.2f} bits")
    ax6.axvline(1.5,             color='green', linestyle=':',  label="Target 1.5 bits")
    ax6.set_title("Policy Entropy (real market states)")
    ax6.set_xlabel("Bits")
    ax6.legend(fontsize=8)

    # --- Panel 7: Conviction distribution on REAL states ---
    ax7 = fig.add_subplot(gs[1, 3])
    max_probs = probs_np.max(axis=1)
    ax7.hist(max_probs, bins=40, color='steelblue', edgecolor='black')
    ax7.axvline(MIN_CONVICTION, color='red',   linestyle='--', label=f"Filter {MIN_CONVICTION:.0%}")
    ax7.axvline(max_probs.mean(),color='gold', linestyle='-',  label=f"Mean {max_probs.mean():.1%}")
    ax7.set_title("Conviction Distribution (real states)")
    ax7.set_xlabel("Max action probability")
    ax7.legend(fontsize=8)

    # --- Panel 8: Per-pace sensitivity heatmap ---
    ax8 = fig.add_subplot(gs[2, :2])
    # Sum the end-to-end sensitivities across the final latent units for each input feature
    sensitivities = np.sum(end_to_end_repr_sensitivity, axis=1)
    ind_names = ["RSI", "MACD", "BB", "OBV", "ATR", "MeanDev"]
    pace_labels = [f"pace={p}" for p in PACES]
    grid = np.zeros((NUM_IND, len(PACES)))
    for i in range(NUM_IND):
        for p_idx in range(len(PACES)):
            base = p_idx * 18 + i
            idxs = [base, base + 6, base + 12]
            grid[i, p_idx] = np.mean(sensitivities[idxs])
    im8 = ax8.imshow(grid, aspect='auto', cmap='YlOrRd')
    ax8.set_yticks(range(NUM_IND))
    ax8.set_yticklabels(ind_names)
    ax8.set_xticks(range(len(PACES)))
    ax8.set_xticklabels(pace_labels)
    ax8.set_title("Feature Sensitivity by Indicator × Pace\n(brighter = more influential)")
    fig.colorbar(im8, ax=ax8, shrink=0.8)
    for i in range(NUM_IND):
        for j in range(len(PACES)):
            ax8.text(j, i, f"{grid[i,j]:.3f}", ha='center', va='center', fontsize=7)

    # --- Panel 9: Action distribution bar chart on real states ---
    ax9 = fig.add_subplot(gs[2, 2])
    actions     = np.argmax(probs_np, axis=1)
    counts      = [np.sum(actions == i) / len(actions) for i in range(4)]
    colors      = ['green', 'red', 'orange', 'grey']
    bars        = ax9.bar(ACTION_NAMES, counts, color=colors, edgecolor='black')
    for bar, pct in zip(bars, counts):
        ax9.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"{pct:.1%}", ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax9.set_ylim(0, 1.0)
    ax9.set_title("Action Distribution (real states)")
    ax9.set_ylabel("Fraction of samples")
    ax9.axhline(0.25, color='black', linestyle=':', linewidth=0.8, label="Uniform 25%")
    ax9.legend(fontsize=8)

    # --- Panel 10: Reward head calibration ---
    ax10 = fig.add_subplot(gs[2, 3])
    test_z  = model.get_initial_state(real_batch)
    _, pred_r = model.simulate_next(test_z, cp.full((len(real_states_np), 1), 1.0, dtype=cp.float32))
    pred_r_np = cp.asnumpy(pred_r).flatten()
    ax10.scatter(pred_r_np[:200], np.zeros(200), alpha=0.3, s=10, color='purple')
    ax10.axvline(pred_r_np.mean(), color='red', linestyle='--',
                 label=f"Mean {pred_r_np.mean():.4f}")
    ax10.axvline(0, color='black', linestyle=':')
    ax10.set_title("Dyn_Reward Head Output Range\n(should straddle 0)")
    ax10.set_xlabel("Predicted reward")
    ax10.legend(fontsize=8)

    plt.suptitle("World Model Full Diagnostic", fontsize=15, fontweight='bold', y=1.01)
    os.makedirs("outcomes", exist_ok=True)
    plt.savefig("outcomes/world_model_diagnostic.png", bbox_inches='tight', dpi=120)
    print("📊 Visual diagnostic saved to outcomes/world_model_diagnostic.png")
    plt.show()


# ─────────────────────────────────────────────
# 3. Numerical health check (extended)
# ─────────────────────────────────────────────
def run_numerical_health_check(model, real_states_np, real_returns_np):
    w_repr1     = cp.asnumpy(model.W_repr1)
    w_repr2     = cp.asnumpy(model.W_repr2)
    w_repr3     = cp.asnumpy(model.W_repr3)
    w_dyn_state = cp.asnumpy(model.W_dyn_state)
    w_dyn_reward= cp.asnumpy(model.W_dyn_reward)
    w_pred      = cp.asnumpy(model.W_pred)

    real_batch  = cp.asarray(real_states_np, dtype=cp.float32)
    latents     = model.get_initial_state(real_batch)
    probs_cp, values_cp = model.predict(latents)
    probs_np    = cp.asnumpy(probs_cp)
    values_np   = cp.asnumpy(values_cp).flatten()

    # Calculate end-to-end sensitivity for the representation network
    end_to_end_repr_sensitivity = np.abs(w_repr1) @ np.abs(w_repr2) @ np.abs(w_repr3)

    # ── 1. Sparsity ──────────────────────────────────────────────────────────
    get_sparse = lambda w: np.sum(np.abs(w) < 1e-5) / w.size
    print("\n══════════════════════════════════════════════════")
    print("   🩺  WORLD MODEL HEALTH & BIAS CHECK")
    print("══════════════════════════════════════════════════")
    print(f"\n─── Weight Sparsity ───────────────────────────────")
    print(f"  Repr1     : {get_sparse(w_repr1):.1%}  (target 10-40%)")
    print(f"  Repr2     : {get_sparse(w_repr2):.1%}  (target 10-40%)")
    print(f"  Repr3     : {get_sparse(w_repr3):.1%}  (target 10-40%)")
    print(f"  Dyn_State : {get_sparse(w_dyn_state):.1%}  (target 10-40%)")
    print(f"  Dyn_Reward: {get_sparse(w_dyn_reward):.1%}  (target 10-40%)")
    print(f"  Pred      : {get_sparse(w_pred):.1%}  (target 20-50%)")

    # ── 2. Latent drift ──────────────────────────────────────────────────────
    test_z = model.get_initial_state(
        cp.random.randn(10, INPUT_SIZE).astype(cp.float32))
    mags = [float(cp.linalg.norm(test_z))]
    for _ in range(20):
        test_z, _ = model.simulate_next(test_z, 0.0)
        mags.append(float(cp.linalg.norm(test_z)))
    drift = mags[-1] / (mags[0] + 1e-9)
    status = "✅" if 0.5 < drift < 2.0 else "⚠️ "
    print(f"\n─── Hallucination Drift (20 steps) ────────────────")
    print(f"  {status} Latent norm ratio: {drift:.3f}x  (target 0.5–2.0x)")

    # ── 3. Policy entropy on real states ─────────────────────────────────────
    entropy = -np.sum(probs_np * np.log2(probs_np + 1e-9), axis=1)
    e_mean   = entropy.mean()
    e_median = np.median(entropy)
    pct_collapsed = (entropy < 0.5).mean()
    pct_healthy   = ((entropy >= 1.0) & (entropy <= 1.8)).mean()
    pct_high      = (entropy > 1.8).mean()

    e_status = "✅" if e_mean > 1.5 else ("⚠️ " if e_mean > 0.8 else "🔴")
    print(f"\n─── Policy Entropy (real market states, n={len(probs_np)}) ──")
    print(f"  {e_status} Mean:              {e_mean:.3f} bits  (target >1.5)")
    print(f"     Median:            {e_median:.3f} bits")
    print(f"     Collapsed (<0.5b): {pct_collapsed:.1%}  (want <5%)")
    print(f"     Healthy (1-1.8b):  {pct_healthy:.1%}  (want >50%)")
    print(f"     High (>1.8b):      {pct_high:.1%}  (want <20%)")

    # ── 4. Action distribution + avg conviction ───────────────────────────────
    actions   = np.argmax(probs_np, axis=1)
    max_probs = probs_np.max(axis=1)
    print(f"\n─── Action Distribution + Conviction ──────────────")
    for i, name in enumerate(ACTION_NAMES):
        mask      = actions == i
        count_pct = mask.mean()
        avg_conv  = probs_np[mask, i].mean() if mask.any() else 0.0
        print(f"  {name:<6}: {count_pct:.1%} of samples  |  avg conviction {avg_conv:.1%}")

    # ── 5. Conviction filter effectiveness ────────────────────────────────────
    trade_mask   = (actions == 0) | (actions == 1)  # LONG or SHORT chosen
    above_filter = (max_probs > MIN_CONVICTION)
    tradeable    = (trade_mask & above_filter).mean()
    print(f"\n─── Conviction Filter (threshold={MIN_CONVICTION:.0%}) ──────────")
    print(f"  Mean max prob:           {max_probs.mean():.1%}")
    print(f"  % above filter:          {above_filter.mean():.1%}  (trades that pass)")
    print(f"  % tradeable (L/S+filter):{tradeable:.1%}  (want 10-40%)")
    print(f"  % high conviction (>70%):{(max_probs > 0.70).mean():.1%}")

    # ── 6. Value head accuracy ────────────────────────────────────────────────
    n = min(len(values_np), len(real_returns_np))
    if n > 10 and np.std(values_np[:n]) > 1e-9 and np.std(real_returns_np[:n]) > 1e-9:
        corr = np.corrcoef(values_np[:n], real_returns_np[:n])[0, 1]
        v_status = "✅" if corr > 0.05 else ("⚠️ " if corr > 0.0 else "🔴")
        print(f"\n─── Value Head Accuracy ────────────────────────────")
        print(f"  {v_status} Value vs Actual Return corr: {corr:+.4f}  (target >0.05)")
        print(f"     Value mean: {values_np[:n].mean():.4f}  std: {values_np[:n].std():.4f}")
        print(f"     Return mean:{real_returns_np[:n].mean():.6f}  std: {real_returns_np[:n].std():.6f}")
    else:
        print(f"\n─── Value Head Accuracy ────────────────────────────")
        print(f"  ⚠️  Insufficient data for correlation check")

    # ── 7. Reward head calibration ────────────────────────────────────────────
    _, pred_r = model.simulate_next(latents, cp.full((len(real_states_np), 1), 1.0, dtype=cp.float32))
    pred_r_np = cp.asnumpy(pred_r).flatten()
    print(f"\n─── Reward Head Calibration ────────────────────────")
    print(f"  Predicted reward mean: {pred_r_np.mean():.5f}  (want ≈0)")
    print(f"  Predicted reward std:  {pred_r_np.std():.5f}  (want >0.001)")
    print(f"  Range: [{pred_r_np.min():.4f}, {pred_r_np.max():.4f}]")
    dead = np.abs(pred_r_np).max() < 1e-4
    print(f"  {'🔴 Dead reward head (all near 0)' if dead else '✅ Reward head is active'}")

    # ── 8. MCTS vs raw policy agreement ──────────────────────────────────────
    print(f"\n─── MCTS vs Raw Policy Agreement (n=50) ───────────")
    planner = MCTSPlanner(model, lookahead_depth=10)
    agreements, mcts_actions, raw_actions_list = 0, [], []
    for i in range(50):
        s = real_states_np[i * (len(real_states_np) // 50)]
        raw_probs_cp, _ = model.predict(model.get_initial_state(
            cp.asarray(s, dtype=cp.float32)))
        raw_a = int(cp.argmax(raw_probs_cp))
        mcts_a, _ = planner.search_best_action(s, temp=0.05)
        raw_actions_list.append(raw_a)
        mcts_actions.append(mcts_a)
        if raw_a == mcts_a:
            agreements += 1
    agree_pct = agreements / 50
    a_status  = "✅" if 0.5 < agree_pct < 0.95 else "⚠️ "
    print(f"  {a_status} Agreement rate: {agree_pct:.1%}")
    print(f"     (>95% = MCTS adds nothing | <50% = policy and planner conflict)")
    mcts_dist = [mcts_actions.count(i) / 50 for i in range(4)]
    raw_dist  = [raw_actions_list.count(i) / 50 for i in range(4)]
    print(f"  {'Action':<8}  {'MCTS':>6}  {'RawPi':>6}")
    for i, name in enumerate(ACTION_NAMES):
        print(f"  {name:<8}  {mcts_dist[i]:>6.1%}  {raw_dist[i]:>6.1%}")

    # ── 9. Per-pace sensitivity breakdown ────────────────────────────────────
    # Sum the end-to-end sensitivities across the final latent units for each input feature
    sensitivities = np.sum(end_to_end_repr_sensitivity, axis=1)
    ind_names     = ["RSI", "MACD", "BB", "OBV", "ATR", "MeanDev"]
    pace_values   = list(PACES)
    print(f"\n─── Per-Pace Feature Sensitivity ───────────────────")
    header = f"  {'Indicator':<12}" + "".join(f"  p={p:<4}" for p in pace_values)
    print(header)
    for i, name in enumerate(ind_names):
        row = f"  {name:<12}"
        for p_idx in range(len(PACES)):
            base = p_idx * 18 + i
            idxs = [base, base + 6, base + 12]
            row += f"  {np.mean(sensitivities[idxs]):.3f} "
        print(row)
    port_sens = np.mean(sensitivities[-2:])
    print(f"  {'PORTFOLIO':<12}  {port_sens:.4f}  (position + unrealized PnL)")

    # ── 10. Dead feature detection ────────────────────────────────────────────
    dead_features = np.where(sensitivities < 1e-3)[0]
    print(f"\n─── Dead Feature Detection ─────────────────────────")
    if len(dead_features) > 0:
        pct_dead = len(dead_features) / len(sensitivities)
        print(f"  ⚠️  {len(dead_features)} dead input features ({pct_dead:.1%}) — consider pruning paces")
        # Map back to indicator names for the first 90 features
        for f_idx in dead_features[:10]:  # show first 10
            if f_idx < 90:
                p_idx   = f_idx // 18
                ind_idx = (f_idx % 18) % 6
                state_t = ["raw", "slope", "std"][(f_idx % 18) // 6]
                print(f"     Feature {f_idx:3d}: {ind_names[ind_idx]:<8} pace={pace_values[p_idx]} [{state_t}]")
    else:
        print(f"  ✅ No dead features detected")

    print("\n══════════════════════════════════════════════════\n")


# ─────────────────────────────────────────────
# 4. Main entry point
# ─────────────────────────────────────────────
def analyze_world_model(paces=PACES):
    model = load_model()

    try:
        raw_ind, prices = load_real_data()
        real_states, _, real_returns = build_real_states(raw_ind, prices, n=SAMPLE_N)
        have_real = True
        print(f"✅ Built {len(real_states)} real 92-d state vectors for analysis")
    except FileNotFoundError as e:
        print(f"⚠️  {e}")
        print("⚠️  Falling back to random inputs — results will be less meaningful")
        real_states  = np.random.randn(SAMPLE_N, INPUT_SIZE).astype(np.float32)
        real_returns = np.zeros(SAMPLE_N, dtype=np.float32)
        have_real    = False

    plot_visual_diagnostics(model, real_states)
    run_numerical_health_check(model, real_states, real_returns)

    if not have_real:
        print("⚠️  NOTE: Master CSV not found. All checks used random noise inputs.")
        print("   Run data_manager.update_master_data() first for meaningful results.")


if __name__ == "__main__":
    analyze_world_model()