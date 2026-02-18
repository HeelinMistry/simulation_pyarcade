import arcade
import numpy as np
from arcade.shape_list import ShapeElementList, create_line


class RegressionEnv(arcade.Window):
    def __init__(self, data_df, agents, brain, show_chart=False):
        super().__init__(1000, 700, "Regression Brain Realized P/L Sim")
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

            # 2. AGENT LOGIC (Threshold-based Execution)
            for agent in self.agents:
                if idx % agent.pace == 0:
                    state, action = agent.act(row)

                    # Pass the unrealized profit to the brain for learning
                    agent.handle_reward(state, action, row['Close'])

                    if action in [0, 1]:
                        # Styling: Make the Master Agent stand out if it exists
                        if hasattr(agent, 'name') and "MASTER" in agent.name:
                            color = arcade.color.GOLD if action == 0 else arcade.color.PURPLE
                            symbol = "◈"
                            size = 14
                        else:
                            color = arcade.color.GREEN if action == 0 else arcade.color.RED
                            symbol = "▲" if action == 0 else "▼"
                            size = 8

                        sig_text = arcade.Text(
                            text=symbol,
                            x=self.get_chart_x(idx),
                            y=self.get_chart_y(row['Close']),
                            color=color,
                            font_size=size,
                            anchor_x="center",
                            anchor_y="center"
                        )
                        self.signal_labels.append(sig_text)

            self.current_tick += 1

    def draw_profit_meter(self):
        """Replaces draw_probability_bars to show the continuous regression output."""
        safe_idx = min(self.current_tick, len(self.df) - 1)
        row = self.df.iloc[safe_idx]

        # We sample the state from the first agent to see what the brain is thinking right now
        state = self.agents[0].get_state(row)
        prediction = self.brain.predict(state)

        # UI Positioning
        base_x = 750
        base_y = 100

        # Draw the "Zero" line
        arcade.draw_line(base_x, base_y - 20, base_x, base_y + 20, arcade.color.WHITE, 2)

        # Scale the prediction for visual rendering (e.g., 0.005 profit = 100 pixels)
        pixel_scale = 20000
        bar_length = prediction * pixel_scale

        # Determine color based on whether we are predicting profit or loss
        color = arcade.color.APPLE_GREEN if prediction > 0 else arcade.color.BITTERSWEET

        # Draw the dynamic meter
        if bar_length != 0:
            rect = arcade.rect.XYWH(
                base_x + (bar_length / 2) if bar_length > 0 else base_x - (abs(bar_length) / 2),
                base_y,
                abs(bar_length),
                20
            )
            arcade.draw_rect_filled(rect, color)

        # Draw the Buy Threshold Marker (assuming agents use ~0.005)
        threshold = 0.005
        threshold_px = threshold * pixel_scale
        arcade.draw_line(base_x + threshold_px, base_y - 15, base_x + threshold_px, base_y + 15, arcade.color.YELLOW, 1)
        arcade.draw_text("BUY TRIGGER", base_x + threshold_px - 30, base_y + 25, arcade.color.YELLOW, 8)

        # Display raw prediction text
        arcade.draw_text(f"Exp. Profit: {prediction:+.3%}", base_x - 40, base_y - 40, color, 12, bold=True)

    def on_draw(self):
        self.clear()

        safe_idx = min(self.current_tick, len(self.df) - 1)
        row = self.df.iloc[safe_idx]

        if self.show_chart:
            self.chart_shapes.draw()
            for label in self.signal_labels:
                label.draw()

        # Update Header
        self.title_text.text = f"Tick: {self.current_tick} | XRP: ${row['Close']:.2f}"
        self.title_text.draw()

        # Update Agent Labels (Now using total_realized_pl instead of total_reward)
        for i, agent in enumerate(self.agents):
            status = "LONG" if len(agent.inventory) > 0 else "FLAT"
            color = arcade.color.GREEN if len(agent.inventory) > 0 else arcade.color.WHITE

            # Fallback to total_reward if total_realized_pl isn't set
            pl = getattr(agent, 'total_realized_pl', getattr(agent, 'total_reward', 0.0))

            self.agent_labels[i].text = f"{agent.name}: {status} | Realized P/L: {pl:.2%}"
            self.agent_labels[i].color = color
            self.agent_labels[i].draw()

        # Replace Entropy with Conviction
        state = self.agents[0].get_state(row)
        prediction = self.brain.predict(state)
        conviction = abs(prediction) * 1000  # Just a scaler for UI readability
        arcade.draw_text(f"Brain Conviction: {conviction:.2f}", 750, 200, arcade.color.WHITE, 10)

        # Draw the new Regression Meter
        self.draw_profit_meter()

    def on_close(self):
        print("Closing window... Saving brain regression weights.")
        self.brain.save()
        super().on_close()