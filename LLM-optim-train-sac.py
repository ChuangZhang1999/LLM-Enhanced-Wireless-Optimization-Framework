import numpy as np
import torch
import gymnasium as gym
import argparse
import os
from env import UAVSecureCommEnv
import utils
import sac
from torch.utils.tensorboard import SummaryWriter
import importlib
import sys


def import_and_reload_module(module_name):
    if module_name in sys.modules:
        del sys.modules[module_name]
    imported_module = importlib.import_module(module_name)
    return imported_module


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, eval_episodes=10, current_timestep=0, log_writer=None):

    eval_env = UAVSecureCommEnv()

    avg_reward = 0.
    avg_secure = 0.
    avg_goal_destination = 0.
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            if type(state) == tuple: state = state[0]
            if type(state) == type({}):
                state = np.concatenate([state['observation'].reshape([1, -1]), state['desired_goal'].reshape([1, -1])],
                                       axis=1)

            action = policy.select_action(revise_state(np.array(state).flatten()).reshape([1, -1]), deterministic=True)
            state, reward, terminated, truncated, infos = eval_env.step(action)

            done = truncated or terminated

            avg_reward += reward
            avg_secure += infos['secrecy_rate']
            if 'goal_reward' in infos:
                avg_goal_destination += infos['goal_reward']

    avg_reward /= eval_episodes
    avg_secure /= eval_episodes
    avg_goal_destination /= eval_episodes
    log_writer.add_scalar("Evaluation/Reward", avg_reward, current_timestep)
    log_writer.add_scalar("Evaluation/Secure", avg_secure, current_timestep)
    log_writer.add_scalar("Evaluation/Goal_Destination", avg_goal_destination, current_timestep)
    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")

    return avg_reward


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="SAC")
    parser.add_argument("--env", default="MountainCarContinuous-v0")  # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=5e3, type=int)  # Time steps initial random policy is used
    parser.add_argument("--eval_freq", default=5e3, type=int)  # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=4e5, type=int)  # Max time steps to run environment
    parser.add_argument("--batch_size", default=256, type=int)  # Batch size for both actor and critic
    parser.add_argument("--hidden_width", type=int, default=256,
                        help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--lr_a", type=float, default=3e-4, help="Learning rate of actor")
    parser.add_argument("--lr_c", type=float, default=3e-4, help="Learning rate of critic")
    parser.add_argument("--discount", default=0.99, type=float)  # Discount factor
    parser.add_argument("--tau", default=0.005, type=float)  # Target network update rate
    parser.add_argument("--adaptive-alpha", action="store_false")
    parser.add_argument("--save_model", action="store_true")  # Save model and optimizer parameters
    parser.add_argument("--load_model", default="")  # Model load file name, "" doesn't load, "default" uses file_name


    parser.add_argument("--revise_path", default="", type=str)  # path of the revise function
    parser.add_argument("--version", default="", type=str)  # path of the revise function

    parser.add_argument("--sid_result_path", default="", type=str)  # path of the revise function
    parser.add_argument("--intrinsic_w", default=0.001, type=float)  # intrinsic_reward_w
    parser.add_argument("--eval", default=0, type=int)  # intrinsic_reward_w

    args = parser.parse_args()
    args.seed = 0
    args.save_model = False

    file_name = f"{args.version}_seed{args.seed}_sac"
    writer = SummaryWriter(log_dir=f"runs/{file_name}")
    if not os.path.exists("./results"):
        os.makedirs("./results")

    if args.save_model and not os.path.exists("./models"):
        os.makedirs("./models")

    env = UAVSecureCommEnv()

    # Set seeds
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    #####################################################################################################
    ########################### set state dim and function##########################################
    if args.eval == 1:
        args.revise_path = f'resources.run-v{args.version}-{args.env}.best_result.v{args.version}-best-{args.env}'
        print('#' * 20)
        print('Evaluate Stage:', args.revise_path)
        print('#' * 20)
    if type(env.observation_space) == gym.spaces.Dict:
        src_state_dim = env.observation_space['observation'].shape[0] + env.observation_space['desired_goal'].shape[0]
    else:
        src_state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    revise_state = import_and_reload_module(args.revise_path).revise_state
    intrinsic_reward = import_and_reload_module(args.revise_path).intrinsic_reward

    test_state = np.zeros([src_state_dim, ])
    state_dim = revise_state(test_state).shape[0]
    #####################################################################################################
    #####################################################################################################

    print("-----------------------------------------------")
    print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}, State Dim: {state_dim}")
    print("-----------------------------------------------")

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": args.discount,
        "tau": args.tau,
    }

    # Initialize policy
    if args.policy == "SAC":
        policy = sac.SAC(state_dim, action_dim, max_action, args.discount, args.tau, args.adaptive_alpha,
                         args.batch_size, args.lr_a, args.lr_c, args.hidden_width)

    if args.load_model != "":
        policy_file = file_name if args.load_model == "default" else args.load_model
        policy.load(f"./models/{policy_file}")

    replay_buffer = utils.ReplayBuffer(state_dim, action_dim)

    # Evaluate untrained policy
    t = 0
    evaluations = [eval_policy(policy, args.env, args.seed, current_timestep=0, log_writer=writer)]
    evaluations_steps = [[evaluations[-1], 0]]
    intrinsic_ratio = []

    state, done = env.reset(), False

    if type(state) == tuple: state = state[0]
    if type(state) == type({}):
        state = np.concatenate([state['observation'].reshape([1, -1]), state['desired_goal'].reshape([1, -1])], axis=1)

    episode_reward, episode_intrinsic_reward = 0, 0
    episode_timesteps = 0
    episode_num = 0
    episode_all_state, episode_all_reward = [], []

    for t in range(int(args.max_timesteps)):

        episode_timesteps += 1

        # Select action randomly or according to policy
        if t < args.start_timesteps:
            action = env.action_space.sample()
        else:
            action = policy.select_action(revise_state(np.array(state).flatten()).reshape([1, -1]))

        # Perform action
        next_state, reward, terminated, truncated, info = env.step(action)

        ### save state and reward ###
        episode_all_state.append(revise_state(np.array(state).flatten()).reshape([-1, 1]))
        episode_all_reward.append(reward)

        if type(next_state) == tuple: next_state = next_state[0]
        if type(next_state) == type({}):
            next_state = np.concatenate(
                [next_state['observation'].reshape([1, -1]), next_state['desired_goal'].reshape([1, -1])], axis=1)
        done = truncated or terminated

        done_bool = float(done)

        # Store data in replay buffer
        intrinsic_r = args.intrinsic_w * intrinsic_reward(revise_state(np.array(state).flatten()))
        if np.isnan(intrinsic_r):
            print(f"[FATAL] NaN in reward, exiting training subprocess.")
            os._exit(1)
        replay_buffer.add(revise_state(np.array(state).flatten()).reshape([1, -1]), action,
                          revise_state(np.array(next_state).flatten()).reshape([1, -1]), reward + intrinsic_r,
                          done_bool)

        state = next_state
        episode_reward += reward
        episode_intrinsic_reward += intrinsic_r

        # Train agent after collecting sufficient data
        if t >= args.start_timesteps:
            policy.learn(replay_buffer)

        if done:
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            print(
                f"Total T: {t + 1} Episode Num: {episode_num + 1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
            writer.add_scalar("Training_Reward/Episode", episode_reward, global_step=t + 1)
            # Reset environment
            state, done = env.reset(), False

            if type(state) == tuple: state = state[0]
            ### gymnasium-robotics ###
            if type(state) == type({}):
                state = np.concatenate([state['observation'].reshape([1, -1]), state['desired_goal'].reshape([1, -1])],
                                       axis=1)

            intrinsic_ratio.append(abs(episode_intrinsic_reward) / (abs(episode_reward) + 1e-5))

            episode_reward, episode_intrinsic_reward = 0, 0
            episode_timesteps = 0
            episode_num += 1

        # Evaluate episode
        if (t + 1) % args.eval_freq == 0:
            evaluations.append(eval_policy(policy, args.env, args.seed,current_timestep=t, log_writer=writer))
            evaluations_steps.append([evaluations[-1], t])

            np.save(f"./results/{file_name}", evaluations)
            if args.save_model: policy.save(f"./models/{file_name}")

        ### all over, then save datas to result ###
    evaluations_steps = np.array(evaluations_steps)
    np.save(args.sid_result_path, evaluations_steps)