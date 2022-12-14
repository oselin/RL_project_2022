import sys, os
#sys.path.insert(0, os.path.abspath(".."))
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

success_streak = 0
reward_threshold = 0

def to_numpy(tensor):
    return tensor.cpu().numpy().flatten()

# Policy training function
def train(agent, env, max_episode_steps=1000):
    global success_streak
    # Run actual training        
    reward_sum, timesteps, done, episode_timesteps = 0, 0, False, 0
    # Reset the environment and observe the initial state
    obs = env.reset()

    
    while not done:
        episode_timesteps += 1
        
        # Sample action from policy
        action, act_logprob = agent.get_action(obs)

        # Perform the action on the environment, get new state and reward
        next_obs, reward, done, _ = env.step(to_numpy(action))

        # Store action's outcome (so that the agent can improve its policy)
        if isinstance(agent, PG):
            done_bool = done
            agent.record(obs, act_logprob, reward, done_bool, next_obs)
        elif isinstance(agent, DDPG):
            # ignore the time truncated terminal signal
            done_bool = float(done) if episode_timesteps < max_episode_steps else 0 
            agent.record(obs, action, next_obs, reward, done_bool)
        else: raise ValueError

        # Store total episode reward
        reward_sum += reward
        timesteps += 1

        # update observation
        obs = next_obs.copy()

        

        if reward_sum >= 400: 
            done = 1
            print("Reward reached, stopping...")

    if reward_sum >= reward_threshold:
        success_streak +=1
        print("Minimum satisfied. Counter at", success_streak)
    else:
        if success_streak: print("------STREAK RESET------")
        success_streak = 0
        
    # update the policy after one episode
    info = agent.update()

    # Return stats of training
    info.update({'timesteps': timesteps,
                'ep_reward': reward_sum,})
    return info



# The main function
@hydra.main(config_path='../configs', config_name='bipedalwalker_easy')
def main(cfg):
    # sed seed
    h.set_seed(cfg.seed)
    cfg.run_id = int(time.time())

    # Setting the reward threshold for early stopping
    global reward_threshold
    reward_threshold = cfg.reward_early_stopping

    print(reward_threshold)
    print("\n\nEnvironment:",cfg.env_name)
    # create folders if needed
    if cfg.env_name == 'LunarLander-v2':
        fldr = f'env1_{cfg.env_name}'
    else:
        fldr = f'env2_{cfg.env_name}'

    work_dir = Path().cwd()/'part1'/fldr
    #if cfg.save_model: h.make_dir(work_dir/"model"/f'{cfg.env_name}_params')
    if cfg.save_logging: 
        h.make_dir(work_dir/"logging")
        L = logger.Logger() # create a simple logger to record stats

    # Model filename
    if cfg.model_path == 'default':
        cfg.model_path = work_dir


    # use wandb to store stats; we aren't currently logging anything into wandb during testing
    if cfg.use_wandb and not cfg.testing:
        wandb.init(project="rl_aalto",
                    name=f'{cfg.exp_name}-{cfg.env_name}-{str(cfg.seed)}-{str(cfg.run_id)}',
                    group=f'{cfg.exp_name}-{cfg.env_name}',
                    config=cfg)

    # create a env
    env = create_env(cfg.config_name, cfg.seed)

    if cfg.save_video:
        # During testing, save every episode
        if cfg.testing:
            ep_trigger = 1
            video_path = work_dir/'video'/'test'
        # During training, save every 50th episode
        else:
            ep_trigger = 50
            video_path = work_dir/'video'/'train'
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
    if not cfg.testing: # training


        for ep in range(trng_episodes + 1):
            # collect data and update the policy
            train_info = train(agent, env)

            
            if cfg.use_wandb:
                wandb.log(train_info)
            if cfg.save_logging:
                L.log(**train_info)
            if (not cfg.silent) and (ep % 100 == 0):
                print({"ep": ep, **train_info})
            
            if success_streak >= 14: #early stopping. 14 success in a row
                print("That's enough")
                break

        
        if cfg.save_model:
            agent.save(cfg.model_path, cfg.seed, cfg.iseasy)


# Entry point of the script
if __name__ == "__main__":
    main()


