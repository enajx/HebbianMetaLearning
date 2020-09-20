import gym
import numpy as np

# https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py
# https://stable-baselines3.readthedocs.io/en/master/_modules/stable_baselines3/common/atari_wrappers.html

class ScaledFloatFrame(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=env.observation_space.shape, dtype=np.float32)

    def observation(self, observation):
        # This undoes the memory optimization, use with smaller replay buffers only.
        return np.array(observation).astype(np.float32) / 255.0

class FireEpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        # Check current lives, make loss of life terminal, then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()

        if self.lives > lives > 0:
            # for Qbert sometimes we stay in lives == 0 condtion for a few frames
            # so its important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            # done = True
            obs, _, done, _ = self.env.step(1)
            if done:
                self.env.reset()
            obs, _, done, _ = self.env.step(2)
            if done:
                self.env.reset()

        self.lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset(**kwargs)
        return obs
