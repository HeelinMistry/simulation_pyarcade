import arcade
import numpy as np
import os
from arcade.shape_list import ShapeElementList, create_line
from PIL import Image


class SimulationEnv(arcade.Window):
    def __init__(self, data_df, executor, show_chart=False, title="Unified Trading Simulator"):
        super().__init__(1000, 700, title)
        self.df = data_df
        self.executor = executor
        self.current_tick = 0
        self.show_chart = show_chart
        self.chart_shapes = ShapeElementList()
        self.signal_labels = []
        self.saved_outcome = False

        self.title_text = arcade.Text(
            "", x=20, y=650, color=arcade.color.WHITE, font_size=18, bold=True
        )

    def get_chart_y(self, price):
        min_p = self.df["Close"].min()
        max_p = self.df["Close"].max()
        return 150 + ((price - min_p) / (max_p - min_p + 1e-9)) * 350

    def get_chart_x(self, tick):
        return (tick / len(self.df)) * 900 + 50

    def on_update(self, delta_time):
        if self.current_tick >= len(self.df) - 1:
            if not self.saved_outcome:
                self.save_final_outcome()
            return

        idx = self.current_tick
        row = self.df.iloc[idx]
        next_row = self.df.iloc[idx + 1]

        # --- 1. Draw Price Line ---
        if self.show_chart:
            line = create_line(
                start_x=self.get_chart_x(idx),
                start_y=self.get_chart_y(row["Close"]),
                end_x=self.get_chart_x(idx + 1),
                end_y=self.get_chart_y(next_row["Close"]),
                color=arcade.color.DARK_GRAY,
                line_width=2
            )
            self.chart_shapes.append(line)

        # --- 2. Feature Preparation ---
        indicators = np.array([
            row["RSI_Scaled"],
            row["MACD_Scaled"],
            row["BB_Scaled"],
            row["OBV_Scaled"]
        ], dtype=np.float32)

        # --- 3. Execute Brain Step ---
        action, probs = self.executor.step(
            indicators=indicators,
            price=row["Close"],
            tick=idx
        )

        # --- 4. Draw Signals ---
        if action in (0, 1, 2):
            if action == 0:
                color, symbol = arcade.color.GREEN, "▲"
            elif action == 1:
                color, symbol = arcade.color.RED, "▼"
            else:
                color, symbol = arcade.color.ORANGE, "✘"

            sig = arcade.Text(
                text=symbol,
                x=self.get_chart_x(idx),
                y=self.get_chart_y(row["Close"]),
                color=color,
                font_size=10,
                bold=True,
                anchor_x="center",
                anchor_y="center"
            )
            self.signal_labels.append(sig)

        self.current_tick += 1

    def draw_probability_bars(self):
        probs = self.executor.last_probs
        if probs is None: return
        labels = ["LONG", "SHORT", "CLOSE", "HOLD"]
        colors = [arcade.color.GREEN, arcade.color.RED, arcade.color.ORANGE, arcade.color.GRAY]
        for i, p in enumerate(probs):
            width = max(1, p * 200)
            rect = arcade.rect.XYWH(750 + width / 2, 50 + (i * 35), width, 25)
            arcade.draw_rect_filled(rect, colors[i])
            arcade.draw_text(f"{labels[i]}: {p:.1%}", 660, 45 + (i * 35), arcade.color.WHITE, 10)

    def get_entropy(self):
        p = self.executor.last_probs
        return -np.sum(p * np.log2(p + 1e-9)) if p is not None else 0.0

    def on_draw(self):
        self.clear()
        safe = min(self.current_tick, len(self.df) - 1)
        row = self.df.iloc[safe]
        if self.show_chart:
            self.chart_shapes.draw()
            for s in self.signal_labels: s.draw()
        
        self.title_text.text = f"Tick: {self.current_tick} | Price: ${row['Close']:.4f}"
        self.title_text.draw()

        status = self.executor.get_status()
        arcade.draw_text(f"Position: {status['position']} | P/L: {status['pnl']:.2%}", 20, 600, arcade.color.WHITE, 12)
        arcade.draw_text(f"Policy Entropy: {self.get_entropy():.2f} bits", 750, 200, arcade.color.WHITE, 10)
        self.draw_probability_bars()

    def save_final_outcome(self):
        """Captures the final state of the chart and saves it as a PNG."""
        os.makedirs("outcomes", exist_ok=True)
        filename = f"outcomes/simulation_outcome_{self.executor.name}.png"
        
        # Capture the color buffer
        image = arcade.get_image()
        image.save(filename)
        
        self.saved_outcome = True
        print(f"✅ Simulation outcome saved to: {filename}")

    def on_close(self):
        super().on_close()
