import sys, os

sys.path.insert(0, os.path.abspath("."))
os.environ["MUJOCO_GL"] = "egl" # for mujoco rendering
import time
from pathlib import Path

import torch
import gym
import hydra
import wandb
import warnings
warnings.filterwarnings("ignore", category=UserWarning) 
warnings.filterwarnings("ignore", category=DeprecationWarning) 

from agents.pg_ac import PG
from agents.ddpg import DDPG
from common import helper as h
from common import logger as logger
from make_env import create_env
import numpy as np

success_streak = 0
reward_threshold = 0

def to_numpy(tensor):
    return tensor.cpu().numpy().flatten()


# Function to test a trained policy
@torch.no_grad()
def test(agent, env, num_episode=50):
    total_test_reward = []
    for ep in range(num_episode):
        obs, done= env.reset(), False
        test_reward = 0

        while not done:
            action, _ = agent.get_action(obs, evaluation=True)
            obs, reward, done, info = env.step(to_numpy(action))
            
            test_reward += reward

        total_test_reward.append(test_reward)

    total_test_reward = np.array(total_test_reward)

    return total_test_reward

# The main function                                 bipedalwalker_easy  lunarlander_continuous_medium
@hydra.main(config_path='../configs', config_name='lunarlander_continuous_medium')
def main(cfg):
    # sed seed
    h.set_seed(cfg.seed)
    cfg.run_id = int(time.time())

    print("\n\nEnvironment:",cfg.env_name,"\nTesting over three seeds")
    # create folders if needed
    if cfg.env_name == 'LunarLander-v2':
        fldr = f'env1_{cfg.env_name}'
    else:
        fldr = f'env2_{cfg.env_name}'

    work_dir = Path().cwd()/'part1'/fldr

    # Model filename
    if cfg.model_path == 'default':
        cfg.model_path = work_dir


    # create a env
    env = create_env(cfg.config_name, cfg.seed)

    if cfg.save_video:
        ep_trigger = 1
        video_path = work_dir/'video'/'test'

        env = gym.wrappers.RecordVideo(env, video_path,
                                        episode_trigger=lambda x: x % ep_trigger == 0,
                                        name_prefix=cfg.exp_name) # save video every 50 episode

    state_shape = env.observation_space.shape
    action_dim = env.action_space.shape[0]
    max_action = env.action_space.high[0]

    # init agent
    if cfg.agent_name == "pg_ac":
        agent = PG(state_shape[0], action_dim, cfg.pg_ac_lr, cfg.pg_ac_gamma)
        trng_episodes = cfg.pg_ac_train_episodes
    else: # ddpg
        agent = DDPG(state_shape, action_dim, max_action,
                    cfg.ddpg_lr, cfg.ddpg_gamma, cfg.ddpg_tau, cfg.ddpg_batch_size, cfg.ddpg_buffer_size)
        trng_episodes = cfg.ddpg_train_episodes
   
   
    # TESTING PHASE

    print("Importing model from", cfg.model_path, "...")

    # load model
    files = os.listdir(cfg.model_path)
    algs, seeds = [], []
    for i in files:
        
        data = i.split("_")
        if len(data) > 1: #avoid useless folders
            if not data[0] in algs: algs.append(data[0])
            if not int(data[2]) in seeds: seeds.append(int(data[2]))

    rws = []
    for sd in seeds:
        print("-----------------------------")
        print("Considering seed:",sd,"for 50 tests")
        agent.load(cfg.model_path, sd, cfg.iseasy)
        rw = test(agent, env, cfg.test_num_episode)
        print("Test reward mean:", np.mean(rw))
        print("Test reward std :", np.std(rw),'\n')
        rws.append(rw)

    rws = np.array(rws)
    print("Averaging the performances over 3 seeds")
    print("Mean:", np.mean(rws))
    print("Std :", np.std(rws))


# Entry point of the script
if __name__ == "__main__":
    main()


