# Risky GridWorld with Q-Learning

This project implements an 8×8 Risky GridWorld environment solved using tabular Q-learning. It includes a training notebook, an interactive Pygame interface, saved Q-table policy, and experiment results. The project was developed for CDS524.

## Environment Description

The environment consists of an 8×8 grid where the agent must navigate from the start state at (0,0) to the goal state at (7,7), while avoiding traps and a moving enemy. The agent's state is represented as a tuple of the form `(agent_row, agent_col, enemy_row, enemy_col)`.

- **Walls** block movement.
- **Traps** are terminal states with negative rewards.
- The **Enemy** moves randomly after the agent, and a collision results in the episode ending.

The episode terminates when the agent:
- Reaches the goal
- Falls into a trap
- Collides with the enemy

## Reward Structure

- **Step cost:** -0.1  
- **Goal:** +20  
- **Trap:** -25  
- **Enemy collision:** -25  
- **Invalid move:** -1  

Reward shaping was tested and compared in the experiment logs.

## Q-Learning Algorithm

Tabular Q-learning with ε-greedy exploration is used for training the agent.

## Repository Structure

- `CDS524_Assignment1_LIUZirui.ipynb`: Jupyter notebook for training, evaluation, and plotting results.  
- `CDS524_Assignment1_LIUZirui.py`: Interactive Pygame UI for playing the game in human or AI mode.  
- `qtable.pkl`: Saved Q-table policy used by the AI in the UI.  
- `metrics_risky.csv`, `metrics_v1.csv`, `risky_no_shaping.csv`, `risky_with_shaping.csv`: Logs for experiment results and performance comparison.  
- `rollout.gif`: Visualization of an episode's rollout.
