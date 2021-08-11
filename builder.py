from envs import *
from networks.neural_network import *
from networks.neat.network import NeatNetwork
from learning_strategies import *


def build_env(config, unity_worker_id):
    if config["name"] in ["simple_spread", "waterworld", "multiwalker"]:
        return PettingzooWrapper(config["name"], config["max_step"])
    elif "Unity" in config["name"]:
        if "CollectApple" in config["name"]:
            return UnityCollectAppleWrapper(config["name"], unity_worker_id, config["max_step"])
    elif config["name"] in ["AndOps"]:
        return AndOps()
    else:
        return GymWrapper(config["name"], config["max_step"], config["pomdp"])


def build_network(config):
    if config["name"] == "gym_model":
        return GymEnvModel(
            config["num_state"],
            config["num_action"],
            config["discrete_action"],
            config["gru"],
        )
    if config["name"] == "NeatNetwork":
        return NeatNetwork(
            config["num_state"],
            config["num_action"],
            config["discrete_action"],
            config["init_mu"],
            config["init_std"],
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

    if strategy_cfg["name"] == "simple_evolution":
        strategy = simple_evolution(
            strategy_cfg["init_sigma"],
            strategy_cfg["sigma_decay"],
            strategy_cfg["elite_num"],
            strategy_cfg["offspring_num"],
        )
    elif strategy_cfg["name"] == "openai_es":
        strategy = openai_es(
            strategy_cfg["init_sigma"],
            strategy_cfg["sigma_decay"],
            strategy_cfg["learning_rate"],
            strategy_cfg["offspring_num"],
        )
    elif strategy_cfg["name"] == "simple_genetic":
        strategy = simple_genetic(
            strategy_cfg["init_sigma"],
            strategy_cfg["sigma_decay"],
            strategy_cfg["elite_num"],
            strategy_cfg["offspring_num"],
        )
    elif strategy_cfg["name"] == "neat":
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
