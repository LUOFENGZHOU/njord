#!/usr/bin/env python
# coding=utf-8

from __future__ import division

# Import built-in packages.
import os
import gc
import pandas

#import psutil
import torch
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter

# Import third-party packages.
from njordtoolbox import Window

# Import classes.
from environment import Environment
from replaymemory import ReplayMemory
from train import Trainer

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

# Set the network hyper parameters.
N_FEATURES = 6
LOOK_BACK = 16
N_ACTIONS = 1
EPISODES = 500
START_RANDOM = False
MAX_EPISODE_COUNTER = 3600 * 24 * 2.0 / PERIOD
ACTION_DIM = 1
STATE_DIM = 6
ACTION_MAX = 1.0
MAX_BUFFER = 100000
MAX_TOTAL_REWARD = 300
EPISODE_PLOT = 25

# -------------------------------------------- #
# LOAD USEFULL CLASSES.
# -------------------------------------------- #

# Load the memroy
memory = ReplayMemory(MAX_BUFFER)

# Load the environment.
env = Environment(FILENAME, QUOTE_QTY, TRADE_QTY)

# Load the trainer.
trainer = Trainer(STATE_DIM, ACTION_DIM, ACTION_MAX, memory)

# Load the window.
window = Window(LOOK_BACK)
window.add_norm("#t", method="log_change", ref="close_price_#t")

# Load the tensorboard writer.
writer = SummaryWriter("tensorboard/runs")

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
	env.reset()
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
	mean_reward = []

	# Run the episode.
	while not done:

		# Increment the episode counter.
		counter += 1

		# Increment the global iteration counter.
		global_iter += 1

		# Get the action.
		if phi is not None:
			if (i_episode + 1) % 25 == 0:
				action = trainer.get_exploitation_action(phi)
			else:
				action = trainer.get_exploration_action(phi)
		else:
			action = -1.0

		# Step the environment.
		next_state, done, reward = env.step(action)
		mean_reward.append(reward)

		# Observe new state.
		if not done:
			if next_state is not None:
				window.append(next_state)
				next_phi = window()
			else:
				atlas.clear()
				next_phi = None
		else:
			next_phi = None

		# Store the transition in memory if valid.
		# Torch the reward.
		if phi is not None and next_phi is not None:

			if isinstance(reward, float) and isinstance(action, float):
				tensor_reward = torch.tensor([reward])
				tensor_action = torch.tensor([action])
				memory.push(phi, tensor_action, next_phi, tensor_reward)

		# Move to the next state.
		phi = next_phi

		# dont update if this is validation
		#if (i_episode + 1) % 5 == 0:
		#	continue		

		# perform the optimization.
		if len(memory) > 128:
			trainer.optimize()

		# Check the episode counter to end simulation.
		if counter > MAX_EPISODE_COUNTER:
			done = True

	# check memory consumption and clear memory
	gc.collect()

	# ------------------------------- #
	# --- Display and tensorboard --- #
	# ------------------------------- #

	# Compute the mean and std of the episode rewards.
	reward_mean = env.rewards.mean()
	reward_std = env.rewards.std()

	# Get the memory size.
	memory_size = len(memory)

	# Display usefull user information.
	print("# info episode ..... = {}".format(i_episode))
	print("# memory capacity .. = {}".format(memory_size))
	print("# mean reward ...... = {}".format(reward_mean))
	print("# std reward ....... = {}".format(reward_std))
	print("# global counter ... = {}".format(global_iter))

	# Write to tensorboard
	writer.add_scalar("reward_mean", reward_mean, i_episode)
	writer.add_scalar("reward_std", reward_std, i_episode)

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

