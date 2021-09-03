from copy import deepcopy

from networks.neat.abstracts import BaseNeat


def wrap_agentid(agent_ids: list, network: BaseNeat) -> dict:
    """Generate agent_group which key is id and value is network.

    This function copies same network for each ids.

    Parameters
    ----------
    agent_ids : list
    network : BaseNeat

    Returns
    -------
    dict
        agent_group which key is id and value is network.
    """
    group = {}
    for agent_id in agent_ids:
        group[agent_id] = deepcopy(network)
    return group
