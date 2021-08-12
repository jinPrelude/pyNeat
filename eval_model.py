import logging
import random
import yaml
import argparse
import numpy as np
import torch
import builder
import os
from copy import deepcopy
from moviepy.editor import ImageSequenceClip
from pyvirtualdisplay import Display


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg-path", type=str, default="conf/ant.yaml")
    parser.add_argument("--ckpt-path", type=str)
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--save-gif", action="store_true")
    parser.add_argument("--server-run", action="store_true")
    args = parser.parse_args()

    with open(args.cfg_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        f.close()
    env = builder.build_env(config["env"], args.seed)
    set_seed(args.seed)
    env.seed(args.seed)
    agent_ids = env.get_agent_ids()

    if args.save_gif:
        run_num = args.ckpt_path.split("/")[-3]
        save_dir = f"test_gif/{run_num}/"
        os.makedirs(save_dir)

    if args.server_run:
        display = Display(visible=0, size=(300, 300))
        display.start()

    network = builder.build_network(config["network"])
    network.load_model(args.ckpt_path)
    for i in range(10):
        models = {}
        for agent_id in agent_ids:
            models[agent_id] = deepcopy(network)
            models[agent_id].reset()
        obs = env.reset()

        done = False
        episode_reward = 0
        ep_step = 0
        ep_render_lst = []
        while not done:
            actions = {}
            for k, model in models.items():
                s = obs[k]["state"][np.newaxis, ...]
                actions[k] = model.forward(s)
            obs, r, done, _ = env.step(actions)
            if "Unity" not in env.name:
                rgb_array = env.render(mode="rgb_array")
                if args.save_gif:
                    ep_render_lst.append(rgb_array)
            episode_reward += r
            ep_step += 1
        print("reward: ", episode_reward, "ep_step: ", ep_step)
        if args.save_gif and "Unity" not in env.name:
            clip = ImageSequenceClip(ep_render_lst, fps=30)
            clip.write_gif(save_dir + f"ep_{i}.gif", fps=30)
        del ep_render_lst
    display.stop()


if __name__ == "__main__":
    main()
