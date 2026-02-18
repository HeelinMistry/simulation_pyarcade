import json


def evaluate_and_save(agents, df, initial_capital=10000):
    final_price = df.iloc[-1]['Close']
    initial_price = df.iloc[0]['Close']

    # Calculate Baseline (Buy & Hold)
    btc_bought = initial_capital / initial_price
    baseline_final_value = btc_bought * final_price

    print("\n--- RESULTS & BASELINE COMPARISON ---")
    print(f"Baseline (Buy & Hold) Final Value: ${baseline_final_value:.2f}")

    best_agent = None
    best_value = 0

    for agent in agents:
        final_val = agent.get_portfolio_value(final_price)
        print(f"Agent {agent.name} (Delay: {agent.tick_delay}) Final Value: ${final_val:.2f}")

        if final_val > best_value:
            best_value = final_val
            best_agent = agent

    # Save Best Outcome to a static JSON model configuration
    if best_agent:
        print(f"\nWINNER: {best_agent.name}! Beating baseline: {best_value > baseline_final_value}")

        outcome_data = {
            "best_strategy": best_agent.name,
            "optimal_tick_delay": best_agent.tick_delay,
            "final_portfolio_value": best_value,
            "total_trades_executed": len(best_agent.history)
        }

        with open("outcomes/best_static_model.json", "w") as f:
            json.dump(outcome_data, f, indent=4)
        print("Best outcome saved to outcomes/best_static_model.json")