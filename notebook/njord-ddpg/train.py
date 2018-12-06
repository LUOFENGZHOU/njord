#!/usr/bin/env python
# coding=utf-8

from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import math

import utils
import model

BATCH_SIZE = 128
LEARNING_RATE = 0.001
GAMMA = 0.99
TAU = 0.001

class Trainer:
	"""DDPG Trainer.
	"""

	def __init__(self, state_dim, action_dim, action_lim, ram):
		"""Special method for object initialisation.

		:param state_dim: Dimensions of state.
		:type state_dim: int.
		:param action_dim: Dimension of action.
		:type action_dim: int.
		:param action_lim: Used to limit action in [-action_lim, action_lim].
		:type action_lim: float.
		:param ram: replay memory buffer object.
		:type ram: buffer.
		"""

		# Set the parameters.
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.action_lim = action_lim
		self.ram = ram
		self.iter = 0

		# Set the noise function.
		self.noise = utils.OrnsteinUhlenbeckActionNoise(self.action_dim)

		# Set the actor.
		self.actor = model.Actor(self.state_dim, self.action_dim, self.action_lim)
		self.target_actor = model.Actor(self.state_dim, self.action_dim, self.action_lim)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),LEARNING_RATE)

		# Set the critic.
		self.critic = model.Critic(self.state_dim, self.action_dim)
		self.target_critic = model.Critic(self.state_dim, self.action_dim)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),LEARNING_RATE)

		# Update the actor and critic networks
		self.hard_update(self.target_actor, self.actor)		
		self.hard_update(self.target_critic, self.critic)

		return

	def soft_update(self, target, source, tau):
		"""Copies the parameters from the source to the target network.

			target = ( 1 - TAU ) * target + TAU * source

		:param target: Target network.
		:type target: torch model.
		:param source: Source network.
		:type source: torch model.
		:param tau: soft update parameter.
		:type tau: float.
		"""
		for t, s in zip(target.parameters(), source.parameters()):
			t.data.copy_( t.data * ( 1.0 - tau ) + s.data * tau )
		return

	def hard_update(self, target, source):
		"""Copies the parameters from source to the target network.

		:param target: Target network.
		:type target: torch model.
		:param source: Source network.
		:type source: torch model.
		"""
		for t, s in zip(target.parameters(), source.parameters()):
			t.data.copy_(s.data)
		return

	def get_exploitation_action(self, state):
		"""Returns the action from target actor.

		:param state: the current state.
		:type state: torch tensor.
		
		:return: sampled action.
		:rtype: numpy array.
		"""
		action = self.target_actor.forward(state).detach()
		return action.item()

	def get_exploration_action(self, state):
		"""Returns the actor action with additive noise.

		:param state: the current state.
		:type state: torch tensor.
		
		:return: sampled action.
		:rtype: numpy array.
		"""
		action = self.actor.forward(state).detach()
		new_action = action.item()
		new_action += float(self.noise.sample() * self.action_lim )
		new_action = min(1.0, max(-1.0, new_action))
		return new_action

	def optimize(self):
		"""Samples a random batch from replay memory and performs optimization
		"""

		# Sample from the mini batch.
		batch = self.ram.sample(BATCH_SIZE)
		s1 = torch.cat(batch.state)
		a1 = torch.cat(batch.action)
		r1 = torch.cat(batch.reward)
		s2 = torch.cat(batch.next_state)

		# --- optimize critic --- #

		# Use target actor exploitation policy here for loss evaluation
		a2 = self.target_actor.forward(s2).detach()
		
		next_val = torch.squeeze(self.target_critic.forward(s2, a2).detach())
		
		# y_exp = r + gamma*Q'( s2, pi'(s2))
		y_expected = r1.view(-1) + GAMMA * next_val
		
		# y_pred = Q( s1, a1)
		y_predicted = torch.squeeze(self.critic.forward(s1, a1))

		# compute critic loss, and update the critic
		loss_critic = F.smooth_l1_loss(y_predicted, y_expected)
		self.critic_optimizer.zero_grad()
		loss_critic.backward()
		self.critic_optimizer.step()

		# --- optimize actor --- #

		pred_a1 = self.actor.forward(s1)
		loss_actor = -1 * torch.sum(self.critic.forward(s1, pred_a1))
		self.actor_optimizer.zero_grad()
		loss_actor.backward()
		self.actor_optimizer.step()

		# Soft update the target actor and critic.
		self.soft_update(self.target_actor, self.actor, TAU)
		self.soft_update(self.target_critic, self.critic, TAU)

		return

	def save_models(self, episode):
		"""Saves the target actor and critic models.

		:param episode: the episode number.
		:type episode: int.
		"""

		# Save the target actor.
		filename_actor = "Models/{}_actor.pt".format(episode)
		torch.save(self.target_actor.state_dict(), filename_actor)

		# Save the target critic.
		filename_critic = "Models/{}_critic.pt".format(episode)		
		torch.save(self.target_critic.state_dict(), filename_critic)

		return

	def load_models(self, episode):
		"""Loads the target actor and critic models, and copies them onto actor and critic models
		:param episode: the count of episodes iterated (used to find the file name)
		:return:
		"""
		self.actor.load_state_dict(torch.load('./Models/' + str(episode) + '_actor.pt'))
		self.critic.load_state_dict(torch.load('./Models/' + str(episode) + '_critic.pt'))
		utils.hard_update(self.target_actor, self.actor)
		utils.hard_update(self.target_critic, self.critic)
		
		print("Models loaded succesfully")