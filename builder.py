from typing import Dict
from envs import *
from networks.neat.network import NeatNetwork
from learning_strategies import *
from loops.loops import ESLoop


def build_env(config: dict, unity_worker_id: int):
    if config["name"] in ["simple_spread", "waterworld", "multiwalker"]:
        return PettingzooWrapper(config["name"], config["max_step"])
    elif config["name"] in ["AndOps"]:
        return AndOps()
    # elif "Unity" in config["name"]:
    #     if "CollectApple" in config["name"]:
    #         return UnityCollectAppleWrapper(config["name"], unity_worker_id, config["max_step"])
    else:
        return GymWrapper(config["name"], config["max_step"], config["pomdp"])


def build_network(config):
    if config["name"] == "NeatNetwork":
        return NeatNetwork(
            config["num_state"],
            config["num_action"],
            config["discrete_action"],
            config["init_mu"],
            config["init_std"],
            config["mutate_std"],
            config["max_weight"],
            config["min_weight"],
            config["probs"],
        )


def build_loop(
    config,
    network,
    agent_ids,
    env_name,
    gen_num,
    n_workers,
    eval_ep_num,
    log,
    save_model_period,
):
    strategy_cfg = config["strategy"]

    if strategy_cfg["name"] == "neat":
        strategy = Neat(
            strategy_cfg["offspring_num"],
            strategy_cfg["crossover_ratio"],
            strategy_cfg["champions_num"],
            strategy_cfg["survival_ratio"],
            strategy_cfg["c1"],
            strategy_cfg["c3"],
            strategy_cfg["delta_threshold"],
        )

    return ESLoop(
        config,
        strategy,
        agent_ids,
        env_name,
        network,
        gen_num,
        n_workers,
        eval_ep_num,
        log,
        save_model_period,
    )
