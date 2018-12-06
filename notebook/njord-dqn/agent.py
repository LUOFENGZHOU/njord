#!/usr/bin/env python
# coding=utf-8

# Import built-in packages.
import random

# Import third party packages.
import torch
import torch.nn.functional as F

# Import models.
from model import Model
from memory import Memory

# Set the agent hyper parameters.
GAMMA 			= 0.95
EPS_START 		= 1.0
EPS_END 		= 0.01
EPS_DECAY 		= 10000
BATCH_SIZE 		= 128
LEARNING_RATE 	= 0.00025
POLICY_UPDATE 	= 4
TARGET_UPDATE 	= 1000
CAPACITY 		= 100000

# --------------------------------- #

class Agent:
	"""Class that implements a Deep Q Networks.

	:attr state_dim: the state dimension.
	:type state_dim: int.
	:attr action_dim: the number of possible action to take.
	:type action_dim: int.
	:attr memory: the memory buffer.
	:type memory: replay memory class.
	:attr policy: the policy action network.
	:type policy: Torch module.
	:attr target: the target action network.
	:type target: Torch module.
	:attr optimizer: the learning optimizer.
	:type optimizer: torch optim.
	:attr rewards: list of rewards.
	:type rewards: list<float>.
	:attr qvalues: list of qvalues.
	:type qvalues: list<float>.
	"""

	def __init__(self, state_dim, action_dim):
		"""Special method for class initialisation.

		:param state_dim: the input state dimension for the Q network.
		:type state_dim: int.
		:param action_dim: the action state dimension for the Q network.
		:type action_dim: int.
		"""

		# Check state and action dimension types.
		if not isinstance(state_dim, int):
			raise TypeError()
		if not isinstance(action_dim, int):
			raise TypeError()

		# Set the state and action dimension.
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.steps = 0
		self.opt_steps = 0
		self.tar_steps = 0

		# Set the memory.
		self.memory = Memory(CAPACITY)

		# Set the networks.
		self.policy = Model(state_dim, action_dim)
		self.target = Model(state_dim, action_dim)
		self.policy.train()
		self.target.eval()
		self.update()

		# Set the records.
		self.rewards = []
		self.qvalues = []

		# Set the optimizer.
		self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=LEARNING_RATE)	

		return

	def __repr__(self):
		"""Special method for class representation.
		"""
		_repr = {
			"steps": self.steps,
			"opt_steps": self.opt_steps,
			"tar_steps": self.tar_steps,
			"epsilon": self.epsilon(),
			"memory": len(self.memory),			
			"rewards": sum(self.rewards)/(len(self.rewards)+1.0E-8),
			"qvalues": sum(self.qvalues)/(len(self.qvalues)+1.0E-8)
		}
		return _repr

	def __str__(self):
		"""Special method for class printable version.
		"""
		_str = []
		for key, item in self.__repr__().items():
			dots = "." * ( 23 - len(key) )
			_str.append("# {} {} : {}".format(key, dots, item))
		return "\n".join(_str)

	# ------------------------------- #
	# ---  0. Agent basic methods --- #
	# ------------------------------- #

	def clear(self):
		"""Clear the current agent.
		"""

		# Reset the agent.
		self.reset()

		# Reset the policy and target networks.
		self.policy = Model(self.state_dim, self.action_dim)
		self.target = Model(self.state_dim, self.action_dim)
		self.policy.train()
		self.target.eval()
		self.update()

		# Reset the optimizer.
		self.optimizer = torch.optim.Adam(self.policy.parameters(), 
			lr=LEARNING_RATE)	

		return

	def load(self, path):
		"""Load agent from path.

		:param path: the name of the path.
		:type path: str.		
		"""
		filename_policy = "{}/{}.txt".format(path, "policy_network")
		filename_target = "{}/{}.txt".format(path, "target_network")
		self.policy.load_state_dict(torch.load(filename_policy))	
		self.target.load_state_dict(torch.load(filename_target))
		return

	def mean_reward(self):
		"""Returns the mean for the rewards.	
		"""
		return sum(self.rewards)/len(self.rewards)

	def mean_qvalue(self):
		"""Returns the mean for the qvalues.
		"""
		return sum(self.qvalues)/len(self.qvalues)

	def reset(self):
		"""Reset the updates.
		"""
		self.rewards.clear()
		self.qvalues.clear()
		return

	def save(self, path):
		"""Save the policy and target networks.

		:param path: the name of the path.
		:type path: str.
		"""
		filename_policy = "{}/{}.txt".format(path, "policy_network")
		filename_target = "{}/{}.txt".format(path, "target_network")
		torch.save(self.policy.state_dict(), filename_policy)
		torch.save(self.target.state_dict(), filename_target)
		return

	def update(self):
		"""Copies the parameters from policy to the target network.
		"""
		for t, p in zip(self.target.parameters(), self.policy.parameters()):
			t.data.copy_(p.data)
		return	

	# ------------------------------- #
	# --- 1. Agent memory methods --- #
	# ------------------------------- #

	def record(self, state, action, next_state, reward):
		"""Records a transition.

		:param state: the state 
		:type sate: torch tensor.
		:param action: the action.
		:type action: int.
		:param next_state: the next state.
		:type next_state: torch tensor.
		:param reward: the reward.
		:type reward: flaot.
		"""

		# Record the reward.
		self.rewards.append(reward)

		# Cast the reward and the action.
		tensor_reward = torch.tensor([reward], dtype=torch.float32)
		tensor_action = torch.tensor([action])

		# Push the transition to the memory.
		self.memory.push(state, tensor_action, next_state, tensor_reward)

		return

	# ------------------------------- #
	# --- 2. Agent action methods --- #
	# ------------------------------- #

	def epsilon(self):
		"""Returns the annealed exploration greedy parameter.

		:return: the epsilon parameter.
		:rtype: float.
		"""
		if self.steps > EPS_DECAY:
			return EPS_END
		else:
			delta = ( EPS_START - EPS_END ) * ( self.steps / EPS_DECAY )
			return EPS_START - delta
	

	def get_exploration_action(self):
		"""Returns an exploration action.

		:return: an action between 0 and action_dim -1
		:rtype: int.
		"""
		return random.randrange(self.action_dim)

	def get_exploitation_action(self, state):
		"""Returns an action that exploits the current network knowledge.

		:param state: the current state.
		:type state: int.

		:return: the Q action.
		:rtype: int.
		:return: teh Q value.
		:rtype: int.
		"""
		with torch.no_grad():
			Q = self.policy(state)
			Q_act = Q.max(1)[1].view(1, 1)
			Q_val = torch.max(Q).item()
		return Q_act, Q_val

	def select_action(self, state, epsilon=None):
		"""Select the action to be taken.

		:param state: the current state.
		:type state: torch tensor.

		:return: the Q action.
		:rtype: int.
		:return: teh Q value.
		:rtype: int.
		"""

		# Increment the number of steps.
		self.steps += 1

		# Get a random seed.
		seed = random.random()

		# Get the current epsilon greedy parameter.
		if epsilon is None:
			epsilon = self.epsilon()

		# Get an exploration/exploitation action.
		if state is not None and seed > epsilon:
			action, qvalue = self.get_exploitation_action(state)
			self.qvalues.append(qvalue)
		else:
			action = self.get_exploration_action()

		return action

	# ------------------------------------- #
	# --- 3. Agent optimisation methods --- #
	# ------------------------------------- #

	def training_batch(self):
		"""Returns the training batch.
		"""
		batch = self.memory.sample(BATCH_SIZE)
		state_batch = torch.cat(batch.state)
		action_batch = torch.stack(batch.action)
		reward_batch = torch.stack(batch.reward)
		return (state_batch, action_batch, reward_batch)

	def optimize(self):
		"""Optimize the policy net for the DQN settings.
		"""

		# Check the size of the memory.
		if len(self.memory) < BATCH_SIZE:
			return

		# Check weather you should update or not.
		if self.steps % POLICY_UPDATE != 0:
			return

		# Step 0.
		# Sample the memory and
		# get the batch states, actions and rewards.
		batch = self.memory.sample(BATCH_SIZE)
		state_batch = torch.cat(batch.state)
		action_batch = torch.cat(batch.action)
		reward_batch = torch.cat(batch.reward)

		# Compute a mask of non-final states and concatenate the batch elements
		non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
		                                      batch.next_state)), dtype=torch.uint8)
		non_final_next_states = torch.cat([s for s in batch.next_state
		                                            if s is not None])
		# Step 1.
		# Compute Q(s_t, a).
		# The policy net computes Q(s_t).
		# We select then the columns of actions taken.
		state_action_values = self.policy(state_batch).gather(1, action_batch)

		# Step 2.
		# Compute V(s_{t+1}) for all next states.
		# The target net computes V(s_{t+1}).
		# We select then the actions with the maximum q values.
		next_state_values = torch.zeros(BATCH_SIZE)
		next_state_values[non_final_mask] = self.target(non_final_next_states).max(1)[0].detach()
		
		# Step 3.
		# Compute the expected Q values.
		expected_state_action_values = reward_batch + GAMMA * next_state_values 

		# Step 4.
		# Compute the Huber loss.
		loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

		# Step 5.
		# Perform one gradient descent step.
		# Clip the gradients smaller than -1 or larger than +1.
		self.optimizer.zero_grad()
		loss.backward()
		for param in self.policy.parameters():
			param.grad.data.clamp_(-1, 1)
		self.optimizer.step()

		# Update the network.
		if self.steps % TARGET_UPDATE == 0:
			self.hard_update()

		return

	def optimize_double(self):
		"""Optimize the policy net for the DDQN settings.
		"""

		# Check the size of the memory.
		if len(self.memory) < BATCH_SIZE:
			return

		# Check weather you should update or not.
		if self.steps % POLICY_UPDATE != 0:
			return

		# Step 0.
		# Sample the memory and
		# get the batch states, actions and rewards.
		batch = self.memory.sample(BATCH_SIZE)
		state_batch  = torch.cat(batch.state)
		action_batch = torch.stack(batch.action)
		reward_batch = torch.stack(batch.reward)

		# Compute a mask of non-final states and concatenate the batch elements
		non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
		                                      batch.next_state)), dtype=torch.uint8)
		non_final_next_states = torch.cat([s for s in batch.next_state
		                                            if s is not None])
		# Step 1.
		# Compute Q(s_t, a).
		# The policy net computes Q(s_t).
		# We select then the columns of actions taken.
		state_action_values = self.policy(state_batch).gather(1, action_batch)

		# Step 2.
		# Compute V(s_{t+1}) for all next states.
		# The target net computes V(s_{t+1}).
		# We select then the actions with the maximum q values.
		next_state_values = torch.zeros(BATCH_SIZE, 1)	
		Q_policy = self.policy(non_final_next_states)

		Q_target = self.target(non_final_next_states)

		Q_argmax = torch.argmax(Q_policy, dim=1, keepdim=True)
		next_state_values[non_final_mask,0] = Q_target.gather(1, Q_argmax)[:,0].detach()
		
		# Step 3.
		# Compute the expected Q values or the targets.
		targets = reward_batch + GAMMA * next_state_values

		# Step 4.
		# Compute the Huber loss.
		loss = F.smooth_l1_loss(state_action_values, targets)

		# Step 5.
		# Perform one gradient descent step.
		# Clip the gradients smaller than -1 or larger than +1.
		self.optimizer.zero_grad()
		loss.backward()
		for param in self.policy.parameters():
			param.grad.data.clamp_(-1, 1)
		self.optimizer.step()

		# Increment the number of optimisation steps taken.
		self.opt_steps += 1

		# Update the network.
		if self.steps % TARGET_UPDATE == 0:
			self.update()
			self.tar_steps += 1

		return
