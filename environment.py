import arcade
import numpy as np
from arcade.shape_list import ShapeElementList, create_line


class SimulationEnv(arcade.Window):
    def __init__(self, data_df, agents, brain, show_chart=False):
        super().__init__(1000, 700, "Long-Only Realized P/L Sim")
        self.df = data_df
        self.agents = agents
        self.brain = brain
        self.current_tick = 0
        self.show_chart = show_chart

        # High-performance batching containers
        self.chart_shapes = ShapeElementList()
        self.signal_labels = []

        self.title_text = arcade.Text(
            text="", x=20, y=650, color=arcade.color.WHITE,
            font_size=18, bold=True
        )

        self.agent_labels = []
        for i, agent in enumerate(self.agents):
            label = arcade.Text("", 20, 600 - (i * 30), arcade.color.WHITE, 12)
            self.agent_labels.append(label)

    def get_chart_y(self, price):
        min_p = self.df['Close'].min()
        max_p = self.df['Close'].max()
        return 150 + ((price - min_p) / (max_p - min_p + 1e-9)) * 350

    def get_chart_x(self, tick):
        return (tick / len(self.df)) * 900 + 50

    def on_update(self, delta_time):
        if self.current_tick < len(self.df) - 1:
            idx = self.current_tick
            row = self.df.iloc[idx]
            next_row = self.df.iloc[idx + 1]

            # 1. BAKE THE LINE
            if self.show_chart:
                line = create_line(
                    start_x=self.get_chart_x(idx),
                    start_y=self.get_chart_y(row['Close']),
                    end_x=self.get_chart_x(idx + 1),
                    end_y=self.get_chart_y(next_row['Close']),
                    color=arcade.color.DARK_GRAY,
                    line_width=2
                )
                self.chart_shapes.append(line)

            # 2. AGENT LOGIC
            for agent in self.agents:
                if idx % agent.pace == 0:
                    state, action = agent.act(row)
                    agent.handle_reward(state, action, row['Close'])

                    if action in [0, 1]:
                        color = arcade.color.GREEN if action == 0 else arcade.color.RED
                        symbol = "▲" if action == 0 else "▼"

                        sig_text = arcade.Text(
                            text=symbol,
                            x=self.get_chart_x(idx),
                            y=self.get_chart_y(row['Close']),
                            color=color,
                            font_size=6,
                            anchor_x="center",
                            anchor_y="center"
                        )
                        self.signal_labels.append(sig_text)

            self.current_tick += 1

    def draw_probability_bars(self):
        safe_idx = min(self.current_tick, len(self.df) - 1)
        row = self.df.iloc[safe_idx]
        state = self.agents[0].get_state(row)
        probs = self.brain.get_probs(state)

        labels = ["BUY", "SELL", "HOLD"]
        colors = [arcade.color.APPLE_GREEN, arcade.color.BITTERSWEET, arcade.color.DIM_GRAY]

        for i, p in enumerate(probs):
            bar_width = p * 200
            rect = arcade.rect.XYWH(750 + (bar_width / 2), 50 + (i * 35), bar_width, 25)
            arcade.draw_rect_filled(rect, colors[i])
            arcade.draw_text(f"{labels[i]}: {p:.1%}", 750 - 90, 50 + (i * 35) - 5, arcade.color.WHITE, 10)

    def on_draw(self):
        self.clear()

        safe_idx = min(self.current_tick, len(self.df) - 1)
        row = self.df.iloc[safe_idx]

        # Respect the show_chart toggle!
        if self.show_chart:
            self.chart_shapes.draw()
            for label in self.signal_labels:
                label.draw()

        # Update Header
        self.title_text.text = f"Tick: {self.current_tick} | XRP: ${row['Close']:.2f}"
        self.title_text.draw()

        # Update Agent Labels
        for i, agent in enumerate(self.agents):
            status = "LONG" if len(agent.inventory) > 0 else "FLAT"
            color = arcade.color.GREEN if len(agent.inventory) > 0 else arcade.color.WHITE
            self.agent_labels[i].text = f"{agent.name}: {status} | P/L: {agent.total_reward:.2%}"
            self.agent_labels[i].color = color
            self.agent_labels[i].draw()

        entropy = self.get_entropy()
        arcade.draw_text(f"Brain Confusion: {entropy:.2f} bits", 750, 200, arcade.color.WHITE, 10)

        self.draw_probability_bars()

    def get_entropy(self):
        safe_idx = min(self.current_tick, len(self.df) - 1)
        row = self.df.iloc[safe_idx]
        state = self.agents[0].get_state(row)
        probs = self.brain.get_probs(state)
        return -np.sum(probs * np.log2(probs + 1e-9))

    def on_close(self):
        print("Closing window... Saving brain weights.")
        self.brain.save()
        super().on_close()