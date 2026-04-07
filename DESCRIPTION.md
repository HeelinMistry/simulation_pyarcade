This project implements a MuZero-inspired World Model architecture designed to trade based on "mental simulations" (hallucinations) of the market. Instead of just predicting if a price goes up or down, it learns the underlying "physics" of the market indicators to plan multiple steps ahead.
🧩 Project Architecture Breakdown
1.
UnifiedWorldModel (agents/UnifiedWorldModel.py): This is the "Brain." It is split into three distinct neural networks (implemented as weight matrices for speed):
◦
Representation: Compresses raw indicators (RSI, MACD, etc.) into a "Latent State" (what the model thinks the market "feels" like right now).
◦
Dynamics: The "Hallucination Engine." Given a Latent State and a hypothetical Action, it predicts the next Latent State and the expected Reward.
◦
Prediction: Evaluates a Latent State to output Policy Probs (the "gut reaction" move) and Value (how "good" this market state is overall).
2.
MCTSPlanner (agents/MCTSPlanner.py): The "Strategist."
◦
It uses the Dynamics network to "look into the future" without actually placing trades.
◦
It simulates all possible actions (LONG, SHORT, etc.) to see which one leads to the highest hallucinated reward and future state value.
◦
Output: It returns the best action and a probability distribution (mcts_probs) used to train the model.
3.
UnifiedExecutor (agents/unified_executor.py): The "Operator."
◦
It manages the State Aggregator, which "warms up" data across multiple timeframes (Paces: 1, 2, 4, 8, 12).
◦
It connects the raw market data to the Planner.
4.
Simulation & Training:
◦
main_headless.py: The training ground. It runs "Stochastic Epochs" where the model explores the data, records its hallucinations vs. reality, and learns from the difference (backpropagation).
◦
live_unified.py: A real-time visualization tool that fetches live Binance data and shows the model's decision-making process in the SimulationEnv UI.
◦
diagnostic.py: A health check for the model. It plots weight sensitivities and tests if the "hallucinations" are stable or if they drift into chaos.