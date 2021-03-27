"""
	Code for collecting failure trajectories using Bayesian Optimization
	Author : Briti Gangopdahyay
	Project : Policy correction using Bayesian Optimization
	Description : The file contains functions for computing failure trajectories given RL policy and
	safety specifications
	Formal Methods Lab, IIT Kharagpur
"""

import sys
import numpy as np
import gym
import GPy
import GPyOpt
from numpy.random import seed
import matplotlib
from eval_policy import choose_best_action
import gym
from network import FeedForwardActorNN, FeedForwardCriticNN

'''
	Bayesian Optimization module for uncovering failure trajectories

	Safety Requirement
	# Requirement 1: We would like the cartpole to not travel more than a certain
	# distance from its original location(2.4) 
	# Always stay within the region (-2.4, 2.4)
'''

#=============================================Global Variables =================================#
policy = None
env = None
traj_spec_dic = {}


'''
	The function called from within the bayesian optimization module
	parameters : bounds containing the sampled variables of the state vector
	return : calls specification function and computes and returns the minimum value
'''
def sample_trajectory(bounds):
	global policy, env, traj_spec_dic
	obs = bounds[0:4]
	max_steps = 400
	env.reset()
	env.env.state = obs
	traj = [obs]
	reward = 0
	iters= 0
	ep_ret = 0
	done = False
	for _ in range(max_steps):
		iters+=1
		action = choose_best_action(obs,policy)
		obs, rew, done, _ = env.step(action)
		#add the observation state to the current trajectory
		traj.append(obs)
		ep_ret += rew
		if done:
			break
	specification_evaluation = safety_spec(traj)
	traj_spec_dic[traj] = specification_evaluation
	print(f'specification_evaluation ========== {specification_evaluation}')
	return specification_evaluation


def run_BO():
	bounds = [(-0.05, 0.05)] * 4 # Bounds on the state
	max_iter = 100
	myProblem = GPyOpt.methods.BayesianOptimization(sample_trajectory,bounds)
	myProblem.run_optimization(max_iter)
	print(myProblem.fx_opt)


#function representing the safety specification
def safety_spec(traj):
	traj = traj[0]
	x_s = np.array(traj).T[0]
	return min(2.4 - np.abs(x_s))


if __name__ == '__main__':
	global policy, env
	env = 

