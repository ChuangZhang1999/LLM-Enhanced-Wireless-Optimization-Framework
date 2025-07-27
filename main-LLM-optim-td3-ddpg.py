import pandas as pd
import numpy as np
import requests
import json
import traceback
import gymnasium as gym
from env import UAVSecureCommEnv
import os
import importlib
import sys
import argparse
import os
import libtmux
import time
import random
import getpass
import shutil
import openai


openai.api_base = "https://api.chatanywhere.tech/v1"


def import_and_reload_module(module_name):
    if module_name in sys.modules:
        del sys.modules[module_name]
    imported_module = importlib.import_module(module_name)
    return imported_module


def init_prompt():
    cur_env = args.env.split('-')[0].lower()
    obs_file = pd.read_excel(args.observation_path, header=None, sheet_name=cur_env)
    content = list(obs_file.iloc[:, 1])
    unit = list(obs_file.iloc[:, -1])
    detail_content = ''
    for ii in range(len(content)):
        detail_content += '- `s[{}]`: '.format(ii) + content[ii] + f' , the unit is {unit[ii]}.' '\n'

    task_description = obs_file.iloc[0, -1]
    total_dim = len(content)

    additional_prompt = ''

    init_prompt_template = f"""
Revise the state representation for a reinforcement learning agent. 
=========================================================
The agent’s task description is:
{task_description}
=========================================================

The current state is represented by a {total_dim}-dimensional Python NumPy array, denoted as `s`.

Details of each dimension in the state `s` are as follows:
{detail_content}
You should design a task-related state representation based on the the source {total_dim} dim to better for reinforcement training, using the detailed information mentioned above to do some caculations, and feel free to do complex caculations, and then concat them to the source state. 

Besides, we want you to design an intrinsic reward function based on the revise_state python function.

That is to say, we will:
1. use your revise_state python function to get an updated state: updated_s = revise_state(s)
2. use your intrinsic reward function to get an intrinsic reward for the task: r = intrinsic_reward(updated_s)
3. to better design the intrinsic_reward, we recommond you use some source dim in the updated_s, which is between updated_s[0] and updated_s[{total_dim - 1}] 
4. however, you must use the extra dim in your given revise_state python function, which is between updated_s[{total_dim}] and the end of updated_s
{additional_prompt}
Your task is to create two Python functions, named `revise_state`, which takes the current state `s` as input and returns an updated state representation, and named `intrinsic_reward`, which takes the updated state `updated_s` as input and returns an intrinsic reward. The functions should be executable and ready for integration into a reinforcement learning environment.

The goal is to better for reinforcement training. Lets think step by step. Below is an illustrative example of the expected output:

```python
import numpy as np
def revise_state(s):
    # Your state revision implementation goes here
    return updated_s
def intrinsic_reward(updated_s):
    # Your intrinsic reward code implementation goes here
    return intrinsic_reward
```
"""
    return init_prompt_template, detail_content


def find_window_and_execute_command(command):
    os.system(command)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="UAVSECURE", type=str)  # OpenAI gym environment name
    parser.add_argument("--max_timesteps", default=4e5, type=int)  # OpenAI gym environment name
    parser.add_argument("--dir", default="resources/", type=str)  # observation description dir
    parser.add_argument("--observation_path", default="env_observation_space.xlsx",
                        type=str)  # observation description path
    parser.add_argument("--sample_count", default=5, type=int)  # how many samples
    parser.add_argument("--train_seed_count", default=1, type=int)  # how many seeds to training
    parser.add_argument("--model", default="gpt-4-1106-preview", type=str)  # which llm should use
    parser.add_argument("--openai_key", default="sk-EkOHUsQF2yF45EHPsJLIRtM71uSVwBVtebvlRGeXNLgmlUyy", type=str)  # openai key
    parser.add_argument("--temperature", default=1.0, type=float)  # init sampling temperature
    # parser.add_argument("--cuda", default=0, type=int)  # which gpu training on
    parser.add_argument('--session_name', action='store', default='0', type=str, help="Session name for run.")
    parser.add_argument('--user_name', default=getpass.getuser(), type=str,
                        help="User's name for window available check.")
    parser.add_argument("--v", default=1, type=int)  # version
    parser.add_argument("--intrinsic_w", default=0.001, type=float)  # init sampling temperature

    args = parser.parse_args()

    gpt_key = args.openai_key
    assert gpt_key != '', 'You should pass an OpenAI Key to get access to GPT.'
    openai.api_key = gpt_key
    temperature = args.temperature

    print('-' * 20)
    print('model: ', args.model)
    print('-' * 20)

    env = UAVSecureCommEnv()
    if type(env.observation_space) == gym.spaces.Dict:
        state_dim = env.observation_space['observation'].shape[0] + env.observation_space['desired_goal'].shape[0]
    else:
        state_dim = env.observation_space.shape[0]

    test_state = np.random.randn(state_dim, )

    prompt, source_state_description = init_prompt()
    dialogs = [
        {
            "role": "user",
            "content": prompt,
        }
    ]

    code_dir = os.path.join(args.dir, f'td3-run-v{args.v}-' + args.env)
    if os.path.exists(code_dir):
        shutil.rmtree(code_dir)
    os.makedirs(code_dir)

    sid_result_dir = os.path.join(code_dir, 'result')
    os.makedirs(sid_result_dir)

    best_result_dir = os.path.join(code_dir, 'best_result')
    os.makedirs(best_result_dir)

    server = libtmux.Server()
    session = server.find_where({"session_name": args.session_name})
    if not session:
        session = server.new_session(session_name=args.session_name)
    assert session



    """
        begin sample state revise function 
    """
    valid_sample_count = 0
    revise_lib_path_buffer, revise_code_buffer, revise_dim_buffer = [], [], []
    assitant_reply, assitant_reward_reply = [0] * args.sample_count, [0] * args.sample_count

    trying_count = 0
    while valid_sample_count != args.sample_count:
        trying_count += 1
        if trying_count == 50:
            print('...Trying Too Much...')
            exit()

        print("---------------------------------------")
        print(f"Current Sample Trying:{valid_sample_count + 1} Trying Count:{trying_count}")
        print("---------------------------------------")

        try:
            params = {'model': args.model, 'messages': dialogs, 'temperature': temperature}
            completion = openai.ChatCompletion.create(**params)
            sid_code = completion['choices'][0]['message']['content']

            assitant_reply[valid_sample_count] = sid_code
            ret_id = sid_code.rindex('return')
            while ret_id < len(sid_code):
                if sid_code[ret_id] == '\n': break
                ret_id += 1
            sid_code = sid_code[sid_code.index('import numpy as np'):ret_id + 1].replace('`', '') + '\n'
            cur_code_path = os.path.join(code_dir, f'sample_{valid_sample_count}.py')
            with open(cur_code_path, 'w') as fp:
                fp.write(sid_code)
                fp.close()

            cur_module = import_and_reload_module(cur_code_path[:-3].replace('/', '.'))
            cur_revise_state_v11 = cur_module.revise_state(test_state)
            cur_revise_dim = cur_revise_state_v11.shape[0]

            cur_intrinsic_reward = cur_module.intrinsic_reward(cur_revise_state_v11)
            assert cur_intrinsic_reward >= -100.0 and cur_intrinsic_reward <= 100.0, cur_intrinsic_reward

            revise_lib_path_buffer.append(cur_code_path[:-3].replace('/', '.'))
            revise_dim_buffer.append(cur_revise_dim)
            revise_code_buffer.append(sid_code)

            valid_sample_count += 1

        except requests.Timeout:
            print("...request timeout...")
        except Exception as e:
            traceback.print_exc()

    """
        begin training using the functions
    """
    for train in range(args.sample_count):
        for seed_try in range(args.train_seed_count):
            cur_version = f'v{args.v}-{args.env}-train{train}_td3'
            cur_sid_result_path = f'{sid_result_dir}/train{train}_s{seed_try}_td3.npy'
            cur_seed = random.randint(0, 100000)
            cur_training_command = f'python LLM-optim-train-td3-ddpg.py --env {args.env} --revise_path {revise_lib_path_buffer[train]} --version {cur_version} --sid_result_path {cur_sid_result_path} --seed {cur_seed} --max_timesteps {int(args.max_timesteps)} --intrinsic_w {args.intrinsic_w}'
            find_window_and_execute_command(cur_training_command)