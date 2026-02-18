from agents.agent_monte_carlo import MCAgent as Agent
from agents.probabilistic_brain import ProbabilisticBrain
from data.data_manager import update_master_data


def run_stochastic_epoch(agents, df, steps_per_agent=2000):
    """
    Each agent teleports to a random spot, takes a fixed number of actions,
    and teleports again if a trade completes.
    """
    for agent in agents:
        # 1. Start each agent at a fresh, random location in the data
        # Ensure there is enough room for the history lookback
        agent.reset_and_teleport(df)

        for _ in range(steps_per_agent):
            # 2. Extract the current row based on the agent's internal index
            row = df.iloc[agent.current_index]

            # 3. Decision & Action
            state, action = agent.act(row)

            # 4. Handle Reward (In MCAgent, this stores memory and learns on SELL)
            # Use 'Close' or whatever your price column is named
            reward = agent.handle_reward(state, action, row['Close'])

            # 5. Teleport Logic (The Stochastic Part)
            # If the agent just SOLD (action 1), it 'finishes' that scenario.
            # We jump it to a new random index to see a different market condition.
            if action == 1 and reward != 0:
                agent.reset_and_teleport(df)
            else:
                # Otherwise, move the index forward by the agent's pace
                agent.current_index += agent.pace

            # 6. Safety Guard: If we hit the end of the data, teleport back
            if agent.current_index >= len(df) - 1:
                agent.reset_and_teleport(df)


def run_sim():
    df = update_master_data()

    # --- 1. Intelligence ---
    brain = ProbabilisticBrain(alpha=0.001)

    agents = [
        Agent("Scalper_15m", pace=1, brain=brain),
        Agent("Scalper_30m", pace=2, brain=brain),
        Agent("Swing_1h", pace=4, brain=brain),
        Agent("Swing_2h", pace=8, brain=brain),
        Agent("Trend_4h", pace=16, brain=brain)
    ]

    # Split data: 80% for training, 20% for evaluation
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    eval_df = df.iloc[split_idx:]

    # --- Training with Patient Early Stopping ---
    best_score = -float('inf')
    patience_counter = 0
    max_patience = 10  # Increased from 5 to 10 to allow for 'recovery'
    min_profit_threshold = 0.01  # Don't stop early until we hit at least +1% profit

    for epoch in range(200):
        run_stochastic_epoch(agents, train_df, steps_per_agent=2000)
        current_score = brain.evaluate_strategy(eval_df)

        print(f"Epoch {epoch + 1} | Eval P/L: {current_score:.2%}")

        # Logic: If it's the best so far, SAVE.
        if current_score > best_score:
            best_score = current_score
            patience_counter = 0
            brain.save()
            print(f"New Best Model! ({current_score:.2%})")
        else:
            # Patience only kicks in if we are already in 'Profit'
            # Or if we've been failing for a very long time
            if current_score > 0 or patience_counter > 15:
                patience_counter += 1
            else:
                # If we are still losing, keep training!
                # We don't want to stop while the brain is still 'negative'
                patience_counter = 0

        if patience_counter >= max_patience and best_score > min_profit_threshold:
            print(f"\n>>> TARGET REACHED & STABILIZED. Stopping.")
            break

    # --- 3. Sequential Validation (The Final Exam) ---
    # We still run one sequential pass at the end to see how it performs on the full timeline.
    print("\n>>> TRAINING COMPLETE. Running Sequential Validation...")
    # ... (Keep your existing validation logic here, but use iloc for speed) ...

    brain.save()
    print("Simulation Complete.")

    for agent in agents:
        agent.total_reward = 0.0
        agent.inventory = []
        agent.previous_unrealized_profit = 0.0

    close_prices = df['Close'].values
    # Pre-calculate the states for the whole DF or convert to list of dicts
    rows = df.to_dict('records')

    print("\n>>> Running Efficient Sequential Validation...")

    # 2. Iterate by index only
    for idx in range(len(rows)):
        current_row = rows[idx]
        current_close = close_prices[idx]

        for agent in agents:
            # Check if it's time for this agent to act based on its pace
            if idx % agent.pace == 0:
                state, action = agent.act(current_row)
                # In validation, we just want the reward/stats, no weights updates usually
                agent.handle_reward(state, action, current_close)

    # --- 6. Final Report & Persistence ---

    total_pl = sum(a.total_reward for a in agents)
    print("\n" + "=" * 34)
    print("       FINAL PERFORMANCE       ")
    print("=" * 34)
    for a in agents:
        color = "+" if a.total_reward > 0 else ""
        print(f"{a.name:12} | P/L: {color}{a.total_reward:>7.2%}")
    print("-" * 34)
    print(f"Group Combined Return: {total_pl:.2%}")
    print("Weights & History saved to outcomes/prob_weights.pkl")
    print("=" * 34)
    print("\nRun 'python diagnostic.py' to see the visual brain analysis.")


if __name__ == "__main__":
    run_sim()