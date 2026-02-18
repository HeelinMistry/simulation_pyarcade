import arcade

from agents.agent_long import LongOnlyAgent
from agents.probabilistic_brain import ProbabilisticBrain
from data.data_manager import update_master_data
from environment import SimulationEnv


def run_sim():
    df = update_master_data()

    # --- 3. Intelligence & Agents ---
    # Brain handles the shared Weight Matrix [4x3]
    brain = ProbabilisticBrain(alpha=0.0005)

    # --- THE AGENT SWARM ---
    # We can now add many agents at various paces
    agents = [
        LongOnlyAgent("Scalper_15m", pace=1, brain=brain),
        LongOnlyAgent("Scalper_30m", pace=2, brain=brain),
        LongOnlyAgent("Swing_1h", pace=4, brain=brain),
        LongOnlyAgent("Swing_2h", pace=8, brain=brain),
        LongOnlyAgent("Trend_4h", pace=16, brain=brain),
        LongOnlyAgent("Trend_8h", pace=32, brain=brain)
    ]

    EPOCH_NUM = 20
    print(f"Training Brain: {EPOCH_NUM} Epochs of Deep Learning...")
    for epoch in range(EPOCH_NUM):
        for _, row in df.iterrows():
            for agent in agents:
                if _ % agent.pace == 0:
                    state, action = agent.act(row)
                    agent.handle_reward(state, action, row['Close'])
        print(f"Epoch {epoch + 1}/{EPOCH_NUM} Complete")

    print("Pre-training complete. Resetting Agent stats for Simulation...")
    for agent in agents:
        agent.total_reward = 0.0
        agent.inventory = []
        agent.previous_unrealized_profit = 0.0

    # --- 4. Simulation Execution ---
    print(f"Starting Simulation with {len(agents)} agents...")
    print("Logic: Long-Only | Action Mapping: 0:BUY, 1:SELL, 2:HOLD")

    # SimulationEnv acts as the 'Clock' and 'Visualizer'
    window = SimulationEnv(df, agents, brain)
    arcade.run()

    # --- 5. Final Report ---
    # We aggregate performance from all agents
    total_pl = sum(a.total_reward for a in agents)

    print("\n" + "=" * 34)
    print("       SIMULATION SUMMARY       ")
    print("=" * 34)
    for a in agents:
        print(f"{a.name:12} | Total Reward: {a.total_reward:>7.2%}")
    print("-" * 34)
    print(f"Group Combined Return: {total_pl:.2%}")
    print("Weights saved to outcomes/prob_weights.pkl")
    print("=" * 34)


if __name__ == "__main__":
    run_sim()