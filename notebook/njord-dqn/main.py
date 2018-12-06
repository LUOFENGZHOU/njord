#!/usr/bin/env python
# coding=utf-8

from __future__ import division

# Import built-in packages.
import os
import gc
import time
import pandas

#import psutil
import torch
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter

# Import third-party packages.
from njordtoolbox import Window

# Import classes.
from agent import Agent
from environment import Environment

# Plot options.
plt.ion()
plt.figure()

# -------------------------------------------- #

# Set the hyper parameters.
PLATFORM = "binance"
FILENAME = "dataset/dataset_Binance_1_symbol_BTCUSDT_period_600_dev.csv"
QUOTE_QTY = 1000.0
TRADE_QTY = 100.0
PERIOD = 600
SAVE_PATH = "agent"

# Set the network hyper parameters.
LOOK_BACK = 16
EPISODES = 1000
START_RANDOM = False
MAX_EPISODE_COUNTER = 3600 * 24 * 2.0 / PERIOD
ACTION_DIM = 2
STATE_DIM = 6
ACTION_MAX = 1.0
EPISODE_PLOT = 10

# -------------------------------------------- #
# LOAD USEFULL CLASSES.
# -------------------------------------------- #

# Load the trainer.
agent = Agent(STATE_DIM, ACTION_DIM)

# Load the environment.
env = Environment(FILENAME, QUOTE_QTY, TRADE_QTY)

# Load the window.
window = Window(LOOK_BACK)
window.add_norm("#t", method="pvt_change", ref="close_price_#t")

# Load the tensorboard writer.
timestamp = int(1000 * time.time()) % 1000
writer = SummaryWriter("tensorboard/runs_{}".format(timestamp))

# -------------------------------------------- #
# EPISODES
# -------------------------------------------- #

# Set the global iteration counter.
global_iter = 0

# Loop on the episodes.
for i_episode in range(EPISODES):

	# Display new episode
	print("")
	print("# ---------------------- ")
	print("# START EPISODE = {} ".format(i_episode))
	print("# ---------------------- ")
	print("")

	# Reset the environment and the window.
	env.reset(random=START_RANDOM)
	agent.reset()
	window.clear()

	# Append the initial state to the window.
	state = env.state()
	window.append(state)

	# Set the initial state, 
	# the terminal status and the episode counter.
	phi = None
	done = False
	counter = 0
	prev_action = 0.0

	# Run the episode.
	while not done:

		# Increment the episode counter.
		counter += 1

		# Increment the global iteration counter.
		global_iter += 1

		# Get the action.
		action = agent.select_action(phi)

		# Step the environment.
		next_state, done, reward = env.step(action)

		# Observe new state.
		if not done:
			if next_state is not None:
				window.append(next_state)
				next_phi = window()
			else:
				window.clear()
				next_phi = None
		else:
			next_phi = None

		# Store the transition in memory if valid and move to next state.
		if phi is not None and next_phi is not None:
			agent.record(phi, action, next_phi, reward)
		phi = next_phi

		# Perform the optimization.
		agent.optimize_double()

		# Check the episode counter to end simulation.
		if counter > MAX_EPISODE_COUNTER:
			done = True

	# ------------------------------- #
	# --- Display and tensorboard --- #
	# ------------------------------- #

	# Display usefull user information.
	print(agent.__str__())

	# Write to tensorboard.
	writer.add_scalar("mean_reward", agent.mean_reward(), i_episode)
	writer.add_scalar("mean_qvalue", agent.mean_qvalue(), i_episode)

	# Save the model.
	agent.save(SAVE_PATH)

	# -------------- #
	# --- Figure --- #
	# -------------- #

	if (i_episode % EPISODE_PLOT) == 0:

		fig, ax = plt.subplots(3, sharex=True)

		style = {"linewidth":0.75, "marker":".", "markersize":2.0}

		# plot 0.
		ax[0].plot(
			env.public.records["close_price"].t, 
			env.public.records["close_price"].x, 
			color="r", **style)
		ax[0].set_title("Episode = {}".format(i_episode))
		ax[0].set_ylabel("price")
		ax[0].grid()

		# plot 1.
		ax[1].plot(env.actions.t, env.actions.x)
		ax[1].set_ylabel("actions")		
		ax[1].grid()

		# plot 2.
		ax[2].plot(env.rewards.t, env.rewards.x)
		ax[2].set_ylabel("rewards")
		ax[2].grid()	

		plt.gcf().autofmt_xdate()
		plt.pause(0.1)
		plt.close("all")
