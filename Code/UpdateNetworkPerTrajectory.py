"""
	Code for policy updation once the failure trajectories are obtained via Bayesian Optimization
	Author : Briti Gangopdahyay
	Project : Policy correction using Bayesian Optimization
	Formal Methods Lab, IIT Kharagpur
"""

import gym
import time

import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import MultivariateNormal
from network import FeedForwardActorNN, FeedForwardCriticNN
import sys


#Integrating tensorboard
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

timesteps_per_batch = 4800                 # Number of timesteps to run per batch
max_timesteps_per_episode = 1600           # Max number of timesteps per episode
n_updates_per_iteration = 5                # Number of times to update actor/critic per iteration
# Initialize the covariance matrix used to query the actor for actions
cov_var = torch.full(size=(2,), fill_value=0.5)
cov_mat = torch.diag(cov_var)

logger = {
			't_so_far': 0,          # timesteps so far
			'i_so_far': 0,          # iterations so far
			'batch_lens': [],       # episodic lengths in batch
			'batch_rews': [],       # episodic returns in batch
			'actor_losses': [],     # losses of actor network in current iteration
			'actor_network' : 0		# Actor network
		}

def update_policy(policy, critic, env, observation):
	"""
		The main learning function
	"""
	actor_optim = Adam(policy.parameters(), lr=0.005)
	critic_optim = Adam(critic.parameters(), lr=0.005)
	t_so_far = 0 # Timesteps simulated so far
	i_so_far = 0 # Iterations ran so far
	global logger
	while t_so_far < 1000000:
		# Commence failure trajectory collection based on observation produced by BO
		batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = get_trajectory(policy,env,observation)
		# Calculate how many timesteps we collected this batch
		t_so_far += np.sum(batch_lens)

		# Increment the number of iterations
		i_so_far += 1

		# Logging timesteps so far and iterations so far
		logger['t_so_far'] = t_so_far
		logger['i_so_far'] = i_so_far

		# Calculate advantage at k-th iteration
		V, _ = evaluate(critic, policy, batch_obs, batch_acts)
		A_k = batch_rtgs - V.detach()

		# Same as main PPO code
		A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

		if(len(batch_lens)!=0):
			# This is the loop where we update our network for some n epochs
			for _ in range(n_updates_per_iteration):
				# Calculate V_phi and pi_theta(a_t | s_t)
				V, curr_log_probs = evaluate(critic, policy, batch_obs, batch_acts)
				ratios = torch.exp(curr_log_probs - batch_log_probs)
				surr1 = ratios * A_k
				surr2 = torch.clamp(ratios, 1 - 0.2, 1 + 0.2) * A_k
				actor_loss = (-torch.min(surr1, surr2)).mean()
				critic_loss = nn.MSELoss()(V, batch_rtgs)


				# Calculate gradients and perform backward propagation for actor network
				#Not updating the critic as it has been trained as a baseline
				actor_optim.zero_grad()
				actor_loss.backward(retain_graph=True)
				actor_optim.step()

				logger['actor_losses'].append(actor_loss.detach())
				logger['actor_network'] = policy

			_log_summary()

		torch.save(policy.state_dict(), './ppo_actor_updated.pth')


def get_trajectory(policy,env,observation):
	"""
		This is where we collect the failure trajectories
	"""
	# Batch data
	batch_obs = []
	batch_acts = []
	batch_log_probs = []
	batch_rews = []
	batch_rtgs = []
	batch_lens = []

	ep_rews = []
	t = 0

	#print(f'Environment variables after  modification {env.env.lander.position}')
	render = False
	global logger
	index = 0

	while t < 2048:
		ep_rews = [] # rewards collected per episode
		obs = env.reset()
		obs_ini = obs
		env.env.lander.position[1] = env.env.lander.position[1]+observation
		done = False
		index = index + 1

		for ep_t in range(300):
			# Render environment if specified, off by default
			if render:
				env.render()
			
			t += 1 # Increment timesteps ran this batch so far
			# Track observations in this batch
			batch_obs.append(obs)

			# Calculate action and make a step in the env. 
			# Note that rew is short for reward.
			action, log_prob = get_action(policy,obs)
			obs, rew, done, _ = env.step(action)

			# Track recent reward, action, and action log probability
			ep_rews.append(rew)
			batch_acts.append(action)
			batch_log_probs.append(log_prob)

			# If the environment tells us the episode is terminated, break
			if done:
				break

		# Track episodic lengths and rewards
		batch_lens.append(ep_t + 1)
		batch_rews.append(ep_rews)

		#If a negative reward policy was obtain
		avg_trace_reward = [np.sum(ep_rews) for ep_rews in batch_rews[i]]
		trace_batch_obs = []
		trace_batch_acts = []
		trace_batch_log_probs = []
		trace_batch_rews = []
		trace_batch_lens = []
		obs = obs_ini
		if(avg_trace_reward<0):
			action, log_prob = get_action(policy,obs)
			obs, rew, done, _ = env.step(action)


	#print(f'batch_lens =========={batch_lens} batch_rews ========{batch_rews}')

	number_of_negative_traces = 0
	avg_rtgs = [np.sum(ep_rews) for ep_rews in batch_rews]
	batch_rtgs = compute_rtgs(batch_rews).tolist()

	index_to_remove = []
	for i in range(len(avg_rtgs)):
		# remove the negative traces from collected episode as we only want to update  
		if(avg_rtgs[i] < 0):
			number_of_negative_traces += 1
			#print(f'Number of negative traces {number_of_negative_traces} ====== Avg rtgs for {i} ===== {avg_rtgs[i]}')
			index_to_remove.append(i)

	print(f'Number of negative traces {number_of_negative_traces}')

	for num in reversed(index_to_remove):
		batch_obs.pop(num)
		batch_acts.pop(num)
		batch_log_probs.pop(num)
		batch_rtgs.pop(num)
		batch_lens.pop(num)
		batch_rews.pop(num)

	# Reshape data as tensors in the shape specified in function description, before returning
	batch_obs = torch.tensor(batch_obs, dtype=torch.float)
	batch_acts = torch.tensor(batch_acts, dtype=torch.float)
	batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)
	batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)
	avg_ep_rews = np.mean([np.sum(ep_rews) for ep_rews in batch_rews])


	print(f'batch_lens =========={batch_lens} batch_rews ========{avg_ep_rews}')

	logger['batch_rews'] = batch_rews
	logger['batch_lens'] = batch_lens

	return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens

def _log_summary():
	global logger
	t_so_far = logger['t_so_far']
	i_so_far = logger['i_so_far']
	avg_ep_lens = np.mean(logger['batch_lens'])
	avg_ep_rews = np.mean([np.sum(ep_rews) for ep_rews in logger['batch_rews']])
	avg_actor_loss = np.mean([losses.float().mean() for losses in logger['actor_losses']])
	actor_model = logger['actor_network']

	# Round decimal places for more aesthetic logging messages
	avg_ep_lens = str(round(avg_ep_lens, 2))
	avg_ep_rews = str(round(avg_ep_rews, 2))
	avg_actor_loss = str(round(avg_actor_loss, 5))

	writer.add_scalar("Average Episodic Return", int(float(avg_ep_rews)), t_so_far)
	writer.add_scalar("Average actor Loss", int(float(avg_actor_loss)), t_so_far)

	for name, param in actor_model.named_parameters():
		if 'weight' in name:
			writer.add_histogram(name, param.detach().numpy(), t_so_far)

	# Print logging statements
	print(flush=True)
	print(f"-------------------- Iteration #{i_so_far} --------------------", flush=True)
	print(f"Average Episodic Length: {avg_ep_lens}", flush=True)
	print(f"Average Episodic Return: {avg_ep_rews}", flush=True)
	print(f"Average Loss: {avg_actor_loss}", flush=True)
	print(f"Timesteps So Far: {t_so_far}", flush=True)
	print(f"------------------------------------------------------", flush=True)
	print(flush=True)

	# Reset batch-specific logging data
	logger['batch_lens'] = []
	logger['batch_rews'] = []
	logger['actor_losses'] = []


def compute_rtgs(batch_rews):
	"""
		Compute the Reward-To-Go of each timestep in a batch given the rewards.

		Parameters:
			batch_rews - the rewards in a batch, Shape: (number of episodes, number of timesteps per episode)

		Return:
			batch_rtgs - the rewards to go, Shape: (number of timesteps in batch)
	"""
	# The rewards-to-go (rtg) per episode per batch to return.
	# The shape will be (num timesteps per episode)
	batch_rtgs = []

	# Iterate through each episode
	for ep_rews in reversed(batch_rews):

		discounted_reward = 0 # The discounted reward so far

		# Iterate through all rewards in the episode. We go backwards for smoother calculation of each
		# discounted return (think about why it would be harder starting from the beginning)
		for rew in reversed(ep_rews):
			discounted_reward = rew + discounted_reward * 0.95
			batch_rtgs.insert(0, discounted_reward)

	# Convert the rewards-to-go into a tensor
	batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)

	return batch_rtgs

def get_action(policy, obs):
	global cov_mat
	mean = policy(obs)
	dist = MultivariateNormal(mean, cov_mat)


	# Sample an action from the distribution
	action = dist.sample()

	# Calculate the log probability for that action
	log_prob = dist.log_prob(action)

	# Return the sampled action and the log probability of that action in our distribution
	return action.detach().numpy(), log_prob.detach()


def evaluate(critic, policy, batch_obs, batch_acts):
	"""
		Estimate the values of each observation, and the log probs of
		each action in the most recent batch with the most recent
		iteration of the actor network. Should be called from learn.

		Parameters:
			batch_obs - the observations from the most recently collected batch as a tensor.
						Shape: (number of timesteps in batch, dimension of observation)
			batch_acts - the actions from the most recently collected batch as a tensor.
						Shape: (number of timesteps in batch, dimension of action)

		Return:
			V - the predicted values of batch_obs
			log_probs - the log probabilities of the actions taken in batch_acts given batch_obs
	"""
	# Query critic network for a value V for each batch_obs. Shape of V should be same as batch_rtgs
	global cov_mat
	V = critic(batch_obs).squeeze()

	# Calculate the log probabilities of batch actions using most recent actor network.
	# This segment of code is similar to that in get_action()
	mean = policy(batch_obs)
	dist = MultivariateNormal(mean, cov_mat)
	log_probs = dist.log_prob(batch_acts)

	# Return the value vector V of each observation in the batch
	# and log probabilities log_probs of each action in the batch
	return V, log_probs



def correct_policy(env, actor_model, critic_model, observation):
	"""
		Updates the policy model to correct the failure trajectories.
		Parameters:
			env - the environment to test the policy on
			actor_model - the actor neural network model
			critic_model - critic neural network model
			observation - currently working with one observation to extract the failure trajectory
		Return:
			None
	"""
	print(f"Correcting {actor_model} for observation {observation}", flush=True)

	# If the actor model is not specified, then exit
	if actor_model == '':
		print(f"Didn't specify model file. Exiting.", flush=True)
		sys.exit(0)

	# Extract out dimensions of observation and action spaces
	obs_dim = env.observation_space.shape[0]
	act_dim = env.action_space.shape[0]

	# Build our policy and critic the same way we build our actor model in PPO
	policy = FeedForwardNN(obs_dim, act_dim)
	critic = FeedForwardNN(obs_dim, 1)

	# Load in the actor model saved by the PPO algorithm
	policy.load_state_dict(torch.load(actor_model))
	critic.load_state_dict(torch.load(critic_model))
	observation = 18.12496176

	# Evaluate our policy with a separate module, eval_policy, to demonstrate
	# that once we are done training the model/policy with ppo.py, we no longer need
	# ppo.py since it only contains the training algorithm. The model/policy itself exists
	# independently as a binary file that can be loaded in with torch.
	update_policy(policy=policy, critic=critic, env=env ,observation=observation)
