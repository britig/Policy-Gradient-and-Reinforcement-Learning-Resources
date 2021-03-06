"""
	Code for distance measurement between old policy and updated policy
	Author : Briti Gangopdahyay
	Project : Policy correction using Bayesian Optimization
	Description : The file contains utility functions for distance calculations
	Formal Methods Lab, IIT Kharagpur
"""

import sys
from network import FeedForwardActorNN, FeedForwardCriticNN
import math
import numpy as np
import gym


'''
	Compute distance between the old policy and the updated policy 

	Parameters:
				policy_old (string), policy_new (string), env (open ai gym environment)

	Return:
				Distance metric : Dv(πold||πnew) =1n∑ξi,ξ′i∈ξ12∑ai∈ξi,a′i∈ξ′i|πold(si|ai)−πnew(s′i|a′i)|
'''
def compute_distance(policy_old,policy_new,env):
	#Load the policies
	if policy_old == '':
		print(f"Didn't specify old model file. Exiting.", flush=True)
		sys.exit(0)
	if policy_new == '':
		print(f"Didn't specify new model file. Exiting.", flush=True)
		sys.exit(0)
	obs_dim = env.observation_space.shape[0]
	act_dim = env.action_space.shape[0]
	policy_old = FeedForwardActorNN(obs_dim, act_dim)
	policy_new = FeedForwardActorNN(obs_dim, act_dim)
	trajectory_set_old, trajectory_set_new = collect_trajectories(policy_old,policy_new,env)
	distance = 0
	# One trajectory can be greater than the other
	# traj = {s0,s1.........,s_n}
	for i in range(len(trajectory_set_old)):
		traj_old_i = trajectory_set_old[i]
		traj_new_i = trajectory_set_new[i]
		#print(f'traj_old_i ======== {traj_old_i}')
		#print(f'traj_old_i ======== {traj_new_i}')
		traj_len = min(len(trajectory_set_old[i]),len(trajectory_set_new[i]))
		sub_distance = 0
		for j in range(traj_len):
			sub_distance += (traj_old_i[j]-traj_new_i[j])**2
		#Account for the extra states if any
		if(len(trajectory_set_old[i]) > len(trajectory_set_new[i])):
			for k in range(len(trajectory_set_old[i])):
				sub_distance += (traj_old_i[k]-0)**2
		else:
			for k in range(len(trajectory_set_new[i])):
				sub_distance += (traj_new_i[k]-0)**2


		#print(f'sub_distance =========== {sub_distance}')
		sub_distance = np.sqrt(sub_distance)/2
		distance+= sub_distance
	distance = distance/len(trajectory_set_old)
	distance = sum(distance)/len(distance)
	print(f'distance ========== {distance}')


'''
	Collect trajectories for old policy and the updated policy 

	Parameters:
				policy_old (string), policy_new (string), env (open ai gym environment)

	Return:
				trajectory_set_old
'''
def collect_trajectories(policy_old,policy_new,env):
	# Collect n random trajectories let n=1000 for our experiments 
	trajectory_set_old = []
	trajectory_set_new = []
	t = 0
	print(f'Collecting trajectories for both the policies')
	while t<1000:
		obs_old = env.reset()
		env_state = env
		obs_new = obs_old
		episode_observation_old = []
		episode_observation_new = []
		done_old = False
		done_new = False
		while not done_old:
			#Collecting trajectories for the old policy
			action_old = policy_old(obs_old).detach().numpy()
			obs_old, rew_old, done_old, _ = env.step(action_old)
			episode_observation_old.append(obs_old)

		while not done_new:
			#Collecting trajectories for the updated policy
			action_new = policy_new(obs_new).detach().numpy()
			obs_new, rew_new, done_new, _ = env_state.step(action_new)
			episode_observation_new.append(obs_new)
	
		trajectory_set_old.append(episode_observation_old)
		trajectory_set_new.append(episode_observation_new)
		t = t+1
		#print(f'trajectory_set_old ======== {trajectory_set_old}')

	return trajectory_set_old, trajectory_set_new


if __name__ == "__main__":
	env = gym.make('LunarLanderContinuous-v2')
	compute_distance('ppo_actor_LunarLandar.pth','ppo_actor_updated.pth',env)