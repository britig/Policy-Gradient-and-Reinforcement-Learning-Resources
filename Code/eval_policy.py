"""
	This file is used only to evaluate our trained policy/actor after (Haha not anymore)
	training in main.py with ppo.py. I wrote this file to demonstrate
	that our trained policy exists independently of our learning algorithm,
	which resides in ppo.py. Thus, we can test our trained policy without 
	relying on ppo.py.
"""
import pickle
import Box2D
from Box2D.b2 import * 
from Box2D import *
import numpy as np
from UpdateNetworkPerTrajectory import get_action
import time

def _log_summary(ep_len, ep_ret, ep_num):
		"""
			Print to stdout what we've logged so far in the most recent episode.

			Parameters:
				None

			Return:
				None
		"""
		# Round decimal places for more aesthetic logging messages
		ep_len = str(round(ep_len, 2))
		ep_ret = str(round(ep_ret, 2))

		# Print logging statements
		print(flush=True)
		print(f"-------------------- Episode #{ep_num} --------------------", flush=True)
		print(f"Episodic Length: {ep_len}", flush=True)
		print(f"Episodic Return: {ep_ret}", flush=True)
		print(f"------------------------------------------------------", flush=True)
		print(flush=True)

def rollout(policy, env, render):
	"""
		Returns a generator to roll out each episode given a trained policy and
		environment to test on. 

		Parameters:
			policy - The trained policy to test
			env - The environment to evaluate the policy on
			render - Specifies whether to render or not
		
		Return:
			A generator object rollout, or iterable, which will return the latest
			episodic length and return on each iteration of the generator.

		Note:
			If you're unfamiliar with Python generators, check this out:
				https://wiki.python.org/moin/Generators
			If you're unfamiliar with Python "yield", check this out:
				https://stackoverflow.com/questions/231767/what-does-the-yield-keyword-do
	"""
	# Rollout until user kills process
	while True:
		obs = env.reset()
		#env.env.lander.position[1] = env.env.lander.position[1]+18.12496176
		done = False

		# number of timesteps so far
		t = 0

		# Logging data
		ep_len = 0            # episodic length
		ep_ret = 0            # episodic return

		while not done:
			t += 1

			# Render environment if specified, off by default
			if render:
				env.render()

			# Query deterministic action from policy and run it
			action = policy(obs).detach().numpy()
			print(f'action=========={action}==========observation====={env.env.lander.angle}')
			obs, rew, done, _ = env.step(action)

			# Sum all episodic rewards as we go along
			ep_ret += rew
			
		# Track episodic length
		ep_len = t

		# returns episodic length and return in this iteration
		yield ep_len, ep_ret


def collect_failure_traces(policy, critic, env, render):
	"""
		Returns a generator to roll out each episode given a trained policy and
		environment to test on. 

		Parameters:
			policy - The trained policy to test
			env - The environment to evaluate the policy on
			render - Specifies whether to render or not
		
		Return:
			A generator object rollout, or iterable, which will return the latest
			episodic length and return on each iteration of the generator.

		Note:
			If you're unfamiliar with Python generators, check this out:
				https://wiki.python.org/moin/Generators
			If you're unfamiliar with Python "yield", check this out:
				https://stackoverflow.com/questions/231767/what-does-the-yield-keyword-do
	"""
	# Rollout and correct the trajectory if negative reward is obtained
	while True:
		obs = env.reset()
		obs_ini = obs
		env.env.lander.position[1] = env.env.lander.position[1]+18.12496176
		#Should be able to reproduce the environment state
		env_state = [env.env.lander.position[0],env.env.lander.position[1],env.env.lander.linearVelocity[0],env.env.lander.linearVelocity[1],env.env.lander.angle,env.env.lander.angularVelocity]
		print(f'env state =========={env_state}')
		done = False

		# number of timesteps so far
		t = 0

		# Logging data
		ep_len = 0            # episodic length
		ep_ret = 0            # episodic return
		episode_observation = []
		episode_action = []
		episode_observation.append(obs)

		while not done:
			t += 1

			# Render environment if specified, off by default
			if render:
				env.render()
			# Query deterministic action from policy and run it
			action = policy(obs).detach().numpy()
			obs, rew, done, _ = env.step(action)
			episode_observation.append(obs)
			episode_action.append(action)

			# Sum all episodic rewards as we go along
			ep_ret += rew

		#If the episode terminates with a negative reward
		if ep_ret < 0:
			with open('episode_observation.data', 'wb') as filehandle1:
				# store the observation data as binary data stream
				pickle.dump(episode_observation, filehandle1)

			with open('episode_action.data', 'wb') as filehandle2:
				# store the observation data as binary data stream
				pickle.dump(episode_action, filehandle2)

			with open('env_state.data', 'wb') as filehandle3:
				# store the observation data as binary data stream
				pickle.dump(env_state, filehandle3)

			#Choose a different action than what the policy suggests
			#Hoping that the critic network will give -ve values for bad actions
			for obs in episode_observation:
				action = policy(obs).detach().numpy()
				#get the value from the critic
				value = critic(obs).detach().numpy()
				print(f'value of the critic network ============== {value}')
				print(f'Action given by policy network ============== {action}')

			
		# Track episodic length
		ep_len = t

		# returns episodic length and return in this iteration
		yield ep_len, ep_ret


#Single trajectory displayed
def display(policy,critic,env):
	env.reset()
	with open('episode_observation.data', 'rb') as filehandle1:
		# read episode_observation
		episode_observation = pickle.load(filehandle1)
	with open('episode_action.data', 'rb') as filehandle2:
		# read episode_action
		episode_action = pickle.load(filehandle2)
	with open('env_state.data', 'rb') as filehandle3:
		# read env_state
		env_state = pickle.load(filehandle3)
	#Set the environment to the failure state, this will help in trajectory correction
	print(f'env_state ============= {env_state}')
	env.env.lander.position = b2Vec2(env_state[0],env_state[1]-18.12496176)
	env.env.lander.linearVelocity = b2Vec2(env_state[2],env_state[3])
	env.env.lander.angle = env_state[4]
	env.env.lander.angularVelocity = env_state[5]
	#env.step(np.array([0, 0]))
	#Set the position after step so that the world center does not get changed
	#Took me one whole night to debug
	env.env.lander.position[1] = env.env.lander.position[1]+18.12496176
	print(f'env_state ============= {env.env.lander.position,env.env.lander.linearVelocity,env.env.lander.angle,env.env.lander.angularVelocity}')
	print(f'Environment set')
	ep_ret = 0
	obs = episode_observation[0]
	done = False
	# Commencing trace correction
	prev_val = 0
	action_unsafe = []
	while not done:
		env.render()
		env_state = [env.env.lander.position[0],env.env.lander.position[1],env.env.lander.linearVelocity[0],env.env.lander.linearVelocity[1],env.env.lander.angle,env.env.lander.angularVelocity]
		action = policy(obs).detach().numpy()
		time.sleep(0.1)
		action_unsafe.append(action)
		obs, rew, done, _ = env.step(action)
		curr_val = int(critic(obs).detach().numpy())
		print(f'action ============= {action}========= value ===== {int(curr_val)}')
		print(f'Position ============= {env.env.lander.position}========= env.env.lander.linearVelocity ===== {env.env.lander.linearVelocity} ========= angle ===== {env.env.lander.angle}')
		#Saddle point, this is from where we start correcting the actions 
		if(env.env.lander.angle<-0.92):
			print(f'action unsafe============= {action_unsafe}')
			'''print(f"angle is bad===={env.env.lander.angle}")
			env.render()
			time.sleep(0.1)
			#Reset environment
			env.env.lander.position = b2Vec2(env_state[0],env_state[1])
			env.env.lander.linearVelocity = b2Vec2(env_state[2],env_state[3])
			env.env.lander.angle = env_state[4]
			env.env.lander.angularVelocity = env_state[5]
			#Sample actions such that the current action gives a higher value for the next 
			if env.env.lander.angle<-0.92 or env.env.lander.angle>0.92 :
				action_new = [1,-1]
				print(f'action_new===={action_new}')
			else:
				action_new = [1,1]
			obs, rew, done, _ = env.step(action_new)
			curr_val = int(critic(obs).detach().numpy())
			print(f'action after sampling ============= {action_new}===new angle=={env.env.lander.angle}==== value ===== {int(curr_val)}')
		ep_ret += rew
		prev_val = curr_val'''
	print(f'Reward ============= {rew}')



def eval_policy(policy, env, render=False):
	"""
		The main function to evaluate our policy with. It will iterate a generator object
		"rollout", which will simulate each episode and return the most recent episode's
		length and return. We can then log it right after. And yes, eval_policy will run
		forever until you kill the process. 

		Parameters:
			policy - The trained policy to test, basically another name for our actor model
			env - The environment to test the policy on
			render - Whether we should render our episodes. False by default.

		Return:
			None

		NOTE: To learn more about generators, look at rollout's function description
	"""
	# Rollout with the policy and environment, and log each episode's data
	for ep_num, (ep_len, ep_ret) in enumerate(rollout(policy, env, render)):
		_log_summary(ep_len=ep_len, ep_ret=ep_ret, ep_num=ep_num)

def collect_batches(policy, critic, env, render=False):
	'''for ep_num, (ep_len, ep_ret) in enumerate(collect_failure_traces(policy, critic, env, render)):
		_log_summary(ep_len=ep_len, ep_ret=ep_ret, ep_num=ep_num)
		if(ep_ret<0):
			break'''
	display(policy,critic,env)