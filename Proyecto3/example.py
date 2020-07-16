import gym
from stable_baselines.common.atari_wrappers import make_atari
from stable_baselines.deepq.policies import LnCnnPolicy
from stable_baselines import DQN
import numpy as np

env = gym.make('Boxing-v0')


'''model = DQN(LnCnnPolicy, env, verbose=1,
            double_q=True,
            tensorboard_log='D://Parcial3_Machine')
'''

model1 = DQN.load("deepq_boxing")
model2 = DQN.load("deepq_boxing_dueling")
model3 = DQN.load("deepq_boxing_doubleq")

models = [model1]

#model.learn(total_timesteps=75000)
#model.save("deepq_boxing_doubleq")


j = 1
for model in models:
    reward = []
    for i in range(2):
        done = False
        obs = env.reset()
        episode_reward = 0
        while not done:
            action = env.action_space.sample()
            obs, rewards, done, info = env.step(action)
            episode_reward += rewards
            env.render()
        reward += [episode_reward]
        print(reward)
    print('Recompensa promedio modelo %d: %f' % (j, np.mean(reward).tolist()))
    print('Varianza recompensa modelo %d: %f' % (j, np.var(reward).tolist()))
    j += 1

env.close()
