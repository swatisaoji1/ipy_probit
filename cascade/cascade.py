from __future__ import print_function

import networkx as nx
import random
import numpy
import time
import math
import datetime

"""
1. General Information
Two Tie
a:          probability of becoming aware.(innovation)
b_strong:   probability of becoming aware by word of mouth of strong tie.(imitation)
b_weak:     probability of becoming aware by word of mouth of weak tie.(imitation)
m:          number of node's stong tie neighbor whom adopted
j:          number of node's weak tie neighbor whom adopted
U:          Random number between [0, 1] if U < P(t), node become informed.
P(t) = 1 - (1-a) * ((1-b_strong) ^ m) * ((1-b_weak) ^ j)


One Tie
a:           innovation probability (Innovation)
b:           probability of becoming aware by word of mouth of tie.(Imitation)
m:           number of node's neighbors whom adopted
U:          Random number between [0, 1] if U < P(t), node become informed.
P(t) = 1 - (1-a) * ((1-b) ^ m)


Note:
We use directed graph for cascade model.
Each edge will have a attribute called "type" to mark them as strong tie or weak tie.

2. Definition of neighbors
Two Tie:
strong tie neighbors:   node and neighbor connected by reciprocal edges.
weak tie neighbors:     node and neighbor connected by friends only edges (node following neighbor, but neighbor
                        do not following back.)

One Tie (Type 1):
node and neighbor connected by reciprocal edges.

One Tie (Type 2):
node and neighbor connected by reciprocal edges or friends only edges(node following neighbor, but neighbor do not
following back).
"""

WEAK_TIE = 0
STRONG_TIE = 1
NOT_INFORMED = 0
INFORMED = 1


def main():
    """ This main function is used for test only! """
    return


def get_real_data(g, start_date):
    """
    Get network's weekly aware number start from specific date, how many weeks all nodes become adopted.
    Args:
        g: Directed graph network
        start_date: Start date

    Returns:
        weekly aware number in list
    """
    # Each step take 1 week
    step_time = 7 * 24 * 60 * 60 * 1000

    # Get ref number
    tweet_timestamp = nx.get_node_attributes(g, 'timestamp')

    start_time = int(time.mktime(datetime.datetime.strptime(start_date, "%m/%d/%Y").timetuple())) * 1000

    time_var = start_time + step_time
    last_aware_count = 0
    aware_number = []

    # Establish aware number list
    while tweet_timestamp != {}:
        aware_count = 0
        nodes = tweet_timestamp.keys()
        timestamps = tweet_timestamp.values()
        for node, timestamp in zip(nodes, timestamps):
            if timestamp <= time_var:
                aware_count += 1
                del tweet_timestamp[node]

        time_var += step_time
        last_aware_count += aware_count
        aware_number.append(last_aware_count)

    print('Cost {} steps, awarenumber list is {}'.format(len(aware_number), str(aware_number)))
    return aware_number


def generate_twotie_network(graphml_file, remove_celebrities=False):
    """ Generate graph network from graphml file, add strong/weak ties """
    g = nx.read_graphml(graphml_file)

    if remove_celebrities:
        g = remove_celebrity(g)

    identify_edge_types(g)
    count_ties(g)
    return g


def generate_onetie_network(graphml_file, remove_celebrities=False):
    """ Generate graph network from graphml file, keep only strong ties. """

    g = generate_twotie_network(graphml_file, remove_celebrities=remove_celebrities)
    remove_weak_tie(g)

    return g


def remove_celebrity(g):
    """ Remove celebrity in the socail network.
    celebrity is the node that have followers more than mean + 2 * standard deviation """
    followers = {n: len(g.predecessors(n)) for n in g.nodes()}
    mean = float(sum(followers.values())) / len(followers)
    std = math.sqrt(sum([(n - mean) ** 2 for n in followers.values()]) / float(len(followers)))
    celebrity_threshold = mean + 2 * std            # Threshold is mean + 2 * standard deviation
    print("celebrity threshold is {}".format(celebrity_threshold))
    celebrities = []
    for n in g.nodes():
        if len(g.predecessors(n)) >= celebrity_threshold:
            celebrities.append(n)
    print("Remove {} celebrities, {:.2f}% of whole population.".format(len(celebrities),
                                                                       (float(len(celebrities))/len(followers)) * 100))
    # print(celebrities)
    for n in celebrities:
        g.remove_node(n)
    return g


def generate_cascade_twotie_result(g, a, b_strong, b_weak, stop_steps=0):
    """
    Generate cascade result. The number of informed nodes start from period 0.

    :param g:           graph
    :param a:           innovation probability
    :param b_strong:    probability of becoming aware by word of mouth of strong tie.(imitation)
    :param b_weak:      probability of becoming aware by word of mouth of weak tie.(imitation)
    :param stop_steps:  Indicate when to stop simulation. If stop_steps != 0, return until 95% nodes are informed.
    Otherwise return when stop_steps is reached.

    :return:            list of informed node number for each step
    """
    cascade_data = []
    period0(g)
    period1(g, a)
    steps = 1
    cascade_data.append(get_informed_num(g))

    keep_running = True
    while keep_running:
        steps += 1
        period2_twotie(g, a, b_strong, b_weak)
        informed_nodes = float(get_informed_num(g))
        cascade_data.append(informed_nodes)

        if steps >= stop_steps:
            keep_running = False

    return cascade_data


def generate_cascade_onetie_result(g, a, b, stop_steps):
    """
    Generate cascade result. The number of informed nodes start from period 0.

    :param g:           graph
    :param a:           innovation probability
    :param b:           probability of becoming aware by word of mouth of strong tie.(imitation)
    :param stop_steps:  Indicate when to stop simulation. If stop_steps != 0, return until 95% nodes are informed.
    Otherwise return when stop_steps is reached.

    :return:            list of informed node number for each step
    """
    cascade_data = []
    period0(g)
    period1(g, a)
    steps = 1
    cascade_data.append(get_informed_num(g))

    keep_running = True
    while keep_running:
        steps += 1
        period2_onetie(g, a, b)
        informed_nodes = float(get_informed_num(g))
        cascade_data.append(informed_nodes)

        if steps >= stop_steps:
            keep_running = False

    return cascade_data


def smape(actual_vals, forecast_vals):
    """
    (sMAPE) Symmetric Mean absolute percentage error is  is an accuracy measure based on percentage (or relative) errors
    https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error

    :param actual_vals:  actual value list
    :param forecast_vals: forecast value list
    :return: a float smape value.
    """
    if len(actual_vals) != len(forecast_vals):
        return -1

    smape_sum = 0.0

    for actual, forecast in zip(map(float, actual_vals), map(float, forecast_vals)):
        if actual != 0:
            smape_sum += abs(forecast - actual) / ((abs(forecast) + abs(actual)) / 2)

    return smape_sum / len(actual_vals)


def mape(actual_vals, forecast_vals):
    """"""
    if len(actual_vals) != len(forecast_vals):
        return -1

    mape_sum = 0.0

    for actual, forecast in zip(map(float, actual_vals), map(float, forecast_vals)):
        if actual != 0:
            mape_sum += abs(forecast - actual) / abs(actual)

    return mape_sum / len(actual_vals)


def std(actual_vals, forcast_vals):
    std_sum = 0.0

    for actual, forecast in zip(map(float, actual_vals), map(float, forcast_vals)):
        std_sum += pow((forecast - actual), 2)

    return math.sqrt((std_sum / len(actual_vals)))


def get_informed_num(g):
    """ Return number of informed nodes """
    informed_nodes = sum([1 for n in g.nodes(data=True) if n[1]['informed'] == INFORMED])
    return informed_nodes


def period2_twotie(g, a, b_strong, b_weak):
    """
    Informed nodes begin the word-of-mouth process, if random float u < probability p(t),
     node will moves from non-informed to informed.
     m equal to number of adopted strong tie neighbors.
     j equal to number of adopted weak tie neighbors.
     P(t) = 1 - (1-a) ((1-b_strong) ^ m)((1-b_weak) ^ j)

    :param g:               Graph
    :param a:               innovation probability
    :param b_strong:        imitation probability for strong tie (Word of mouth)
    :param b_weak:          imitation probability for weak tie (Word of mouth)
    """
    informed_nodes = set()
    for n in g.nodes():
        if g.node[n]['informed'] == INFORMED:
            continue  # Skip informed nodes
        else:
            m = 0.0
            j = 0.0

            # Choose out degree only
            neighbors = g.successors(n)

            # Count informed strong/weak tie neighbors
            for neighbor in neighbors:
                if g.node[neighbor]['informed'] == INFORMED:  # Neighbor as Friends
                    if g[n][neighbor]['type'] == WEAK_TIE:
                        j += 1
                    elif g[n][neighbor]['type'] == STRONG_TIE:
                        m += 1
                    else:
                        print("ERROR! Edge type should be strong or weak")

            # Node try become informed.
            p = 1 - (1 - a) * pow((1 - b_weak), j) * pow((1 - b_strong), m)
            u = random.uniform(0, 1)
            if u < p:
                # Record informed nodes, update the nodes at the end of period 2
                informed_nodes.add(n)

    # Update informed nodes in period 2
    for n in informed_nodes:
        if g.node[n]['informed'] != NOT_INFORMED:
            print("ERROR! Node already informed!")
        g.node[n]['informed'] = INFORMED


def period2_onetie(g, a, b):
    """
    Informed nodes begin the word-of-mouth process, if random float u < probability p(t),
     node will moves from non-informed to informed.
     m equal to number of adopted strong tie neighbors.
     j equal to number of adopted weak tie neighbors.
     P(t) = 1 - (1-a) ((1-b) ^ m)

    :param g:               Graph
    :param a:               innovation probability
    :param b:        imitation probability for strong tie (Word of mouth)
    """
    informed_nodes = set()
    for n in g.nodes():
        if g.node[n]['informed'] == INFORMED:
            continue  # Skip informed nodes
        else:
            m = 0.0

            neighbors = g.successors(n)

            # Count informed strong/weak tie neighbors
            for neighbor in neighbors:
                if g.node[neighbor]['informed'] == INFORMED:  # Neighbor as Friends
                    m += 1

            # Node try become informed.
            p = 1 - (1 - a) * pow((1 - b), m)
            u = random.uniform(0, 1)
            if u < p:
                # Record informed nodes, update the nodes at the end of period 2
                informed_nodes.add(n)

    # Update informed nodes in period 2
    for n in informed_nodes:
        if g.node[n]['informed'] != NOT_INFORMED:
            print("ERROR! Node already informed!")
        g.node[n]['informed'] = INFORMED


def period1(g, a):
    """
    The probabilities for each nodes are realized by the following steps.
    For each nodes:
        Step 1: A random number U drawn in the range[0,1]
        Step 2.1: If U < p(t) then the node become informed.
        Step 2.2: If U >= p(t) then the node remain non-informed.
    :param g:  Graph
    :param a:  innovation probability

    Note: Here in period 1, p(t) = a
    """
    for n in g.nodes():
        u = random.uniform(0, 1)
        if u < a:
            g.node[n]['informed'] = INFORMED
        else:
            g.node[n]['informed'] = NOT_INFORMED


def period0(g):
    """
    Add informed property to graph, all nodes are uninformed at the start.
     0: uninformed
     1: informed
    """
    for n in g.nodes():
        g.node[n]['informed'] = NOT_INFORMED


def generate_cascade_twotie_params(a_start, a_end, b_strong_start, b_strong_end, b_weak_start, b_weak_end, precision):
    """
    Generate cascade parameters combination set (a, b_strong, b_weak)
    a in range [a_start, a_end, step=precision)
    b_strong in range [b_strong_start, b_strong_end, step=precision)
    b_weak in range [b_weak_start, b_weak_end, step=precision)

    :return: parameters list

    Note: b_strong > b_weak
    """
    if a_start <= 0:
        a_start = precision

    if b_strong_start <= 0:
        b_strong_start = precision

    if b_weak_start <= 0:
        b_weak_start = precision

    params = []
    for a in numpy.arange(a_start, a_end, precision):
        for b_strong in numpy.arange(b_strong_start, b_strong_end, precision):
            for b_weak in numpy.arange(b_weak_start, b_weak_end, precision):
                if b_strong > b_weak:
                    params.append((a, b_strong, b_weak))
    return params


def generate_cascade_onetie_params(a_start, a_end, b_start, b_end, precision):
    """
    Generate cascade parameters combination set (a, b)
    a in range [a_start, a_end, step=precision)
    b_strong in range [b_start, b_end, step=precision)

    :return: parameters list
    """
    if a_start <= 0:
        a_start = precision

    if b_start <= 0:
        b_start = precision

    params = []

    for a in numpy.arange(a_start, a_end, precision):
        for b in numpy.arange(b_start, b_end, precision):
            params.append((a, b))
    return params


def create_random_graph(population):
    """
    Generate an random graph (For test only)

    :return: a random undirected graph
    """
    g = nx.DiGraph()
    g.add_nodes_from(range(population))

    node_list = []
    for n in g.nodes():
        node_list.append(n)
    if len(node_list) != population:
        print("Node list number do not much!")

    # Generate Edges
    for v in g.nodes():
        num_out_edges = random.randint(3, 15)
        num_in_edges = random.randint(3, 15)

        out_targets = random.sample(node_list, num_out_edges)
        for t in out_targets:
            g.add_edge(v, t)

        in_targets = random.sample(node_list, num_in_edges)
        for s in in_targets:
            g.add_edge(s, v)

    print("Graph have", g.number_of_edges(), "edges. (Should between 18000, 90000)")
    g.remove_edges_from(g.selfloop_edges())
    print("After clean up graph have", g.number_of_edges(), "edges.")

    identify_edge_types(g)
    count_ties(g)

    return g


def count_ties(g):
    """
    Count ties in graph g.

    Args:
        g: Graphml file

    """
    num_strong_edge = 0
    num_weak_edge = 0
    for e in g.edges(data=True):
        if e[2]['type'] == STRONG_TIE:
            num_strong_edge += 1
        elif e[2]['type'] == WEAK_TIE:
            num_weak_edge += 1
        else:
            print("Meet error!")
    print("Graph have {0} edges".format(nx.number_of_edges(g)))
    print("We have {0} strong edges, {1} weak edges.".format(num_strong_edge, num_weak_edge))


def identify_edge_types(g):
    """
    Generate ties in social networks.
    Edge attribute 'type' indicate strong and weak type, 0:weak tie, 1:strong tie.
    Strong tie: Exists edge from target to source and source to target
    Weak tie: Only have edge for 1 direction.

    Args:
        g: Directed graph network

    """
    for n in g.nodes():
        for neighbor in set(g.neighbors(n) + g.predecessors(n)):
            if g.has_edge(n, neighbor) and g.has_edge(neighbor, n):
                g.edge[n][neighbor]['type'] = STRONG_TIE
                g.edge[neighbor][n]['type'] = STRONG_TIE
            elif g.has_edge(n, neighbor):
                g.edge[n][neighbor]['type'] = WEAK_TIE
            elif g.has_edge(neighbor, n):
                g.edge[neighbor][n]['type'] = WEAK_TIE
            else:
                print("Meet error!")


def remove_weak_tie(g):
    """
    Generate one tie in social network. Only keep reciprocal edge. Remove one way edge.
    """
    # Remove weak ties
    for e in g.edges(data=True):
        if e[2]['type'] == WEAK_TIE:
            g.remove_edge(e[0], e[1])
    print("After remove weak tie")
    count_ties(g)


if __name__ == "__main__":
    main()
