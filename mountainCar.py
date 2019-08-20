import gym
import matplotlib.pyplot as plt
import numpy as np
from gym import wrappers
import pickle

pos_space = np.linspace(-1.2, 0.6, 20)
vel_space = np.linspace(-0.07, 0.07, 20)

def main():
	print("\n"*100)
	input("Press enter to start training.")
	run_algorithm()


def plot(mean_rewards):
	plt.plot(mean_rewards)
	plt.savefig('mountaincar.png')
	plt.show()

def create_states(number_of_space_values):
	states = []
	for pos in range(number_of_space_values):
		for vel in range(number_of_space_values):
			states.append((pos, vel))
	return states

def initialize_Q(states):
	action_space = [0, 1, 2]
	Q = {}
	for state in states:
		for action in action_space:
			Q[state, action] = 0
	return Q


def get_state(observation):
	pos, vel =  observation
	pos_bin = int(np.digitize(pos, pos_space))
	vel_bin = int(np.digitize(vel, vel_space))
	return (pos_bin, vel_bin)

def max_action(Q, state, actions=[0, 1, 2]):
	values = np.array([Q[state,a] for a in actions])
	action = np.argmax(values)
	return action

def run_algorithm():
	env = gym.make('MountainCar-v0')
	env._max_episode_steps = 500
	number_of_space_values = 21
	n_games, alpha, gamma, eps = 5000, 0.1, 0.99, 0

	states = create_states(number_of_space_values)
	Q = initialize_Q(states)

	score = 0
	total_rewards = np.zeros(n_games)
	for i in range(n_games):
		done = False
		obs = env.reset()
		state = get_state(obs)
		if i % 500 == 0 and i > 0:
			print('episode ', i, 'score ', score, 'epsilon %.3f' % eps)
		score = 0
		while not done:
			#env.render()
			action = np.random.choice([0,1,2]) if np.random.random() < eps \
					else max_action(Q, state)
			obs_, reward, done, info = env.step(action)
			state_ = get_state(obs_)
			score += reward
			action_ = max_action(Q, state_)
			Q[state, action] = Q[state, action] + \
					alpha*(reward + gamma*Q[state_, action_] - Q[state, action])
			state = state_
		total_rewards[i] = score
		eps = eps - 2/n_games if eps > 0.01 else 0.01
		#print("game: ", i, "score: ", score)

	env.close()

	mean_rewards = np.zeros(n_games)
	for t in range(n_games):
		mean_rewards[t] = np.mean(total_rewards[max(0, t-50):(t+1)])
	plot(mean_rewards)


if __name__ == '__main__':
	main()
