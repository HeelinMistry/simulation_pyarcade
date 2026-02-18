import numpy as np
from matplotlib import pyplot as plt

from agents.probabilistic_brain import ProbabilisticBrain

def analyze_last_run():
    # 1. Create a fresh brain shell
    brain = ProbabilisticBrain()

    # 2. It automatically calls self.load() in __init__,
    # so it's already holding your live run's data!

    print("Logic Map for last saved run:")
    brain.plot_weights()

    brain.plot_decision_regions()
    brain.plot_momentum_logic()

    return brain


if __name__ == "__main__":
    last_brain = analyze_last_run()
