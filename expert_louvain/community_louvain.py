# -*- coding: utf-8 -*-
"""
This module implements community detection.
"""
from __future__ import print_function

import array

import numbers
import warnings

import networkx as nx
import numpy as np

from .community_status import Status

__author__ = """Thomas Aynaud (thomas.aynaud@lip6.fr)"""
#    Copyright (C) 2009 by
#    Thomas Aynaud <thomas.aynaud@lip6.fr>
#    All rights reserved.
#    BSD license.

__PASS_MAX = -1
__MIN = 0.0000001


def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance.

    Parameters
    ----------
    seed : None | int | instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.

    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError("%r cannot be used to seed a numpy.random.RandomState"
                     " instance" % seed)


def partition_at_level(dendrogram, level):
    """Return the partition of the nodes at the given level

    A dendrogram is a tree and each level is a partition of the graph nodes.
    Level 0 is the first partition, which contains the smallest communities,
    and the best is len(dendrogram) - 1.
    The higher the level is, the bigger are the communities

    Parameters
    ----------
    dendrogram : list of dict
       a list of partitions, ie dictionnaries where keys of the i+1 are the
       values of the i.
    level : int
       the level which belongs to [0..len(dendrogram)-1]

    Returns
    -------
    partition : dictionnary
       A dictionary where keys are the nodes and the values are the set it
       belongs to

    Raises
    ------
    KeyError
       If the dendrogram is not well formed or the level is too high

    See Also
    --------
    best_partition : which directly combines partition_at_level and
    generate_dendrogram : to obtain the partition of highest modularity

    Examples
    --------
    >>> G=nx.erdos_renyi_graph(100, 0.01)
    >>> dendrogram = generate_dendrogram(G)
    >>> for level in range(len(dendrogram) - 1) :
    >>>     print("partition at level", level, "is", partition_at_level(dendrogram, level))  # NOQA
    """
    partition = dendrogram[0].copy()
    for index in range(1, level + 1):
        for node, community in partition.items():
            partition[node] = dendrogram[index][community]
    return partition


def modularity(partition, graph, weight='weight'):
    """Compute the modularity of a partition of a graph

    Parameters
    ----------
    partition : dict
       the partition of the nodes, i.e a dictionary where keys are their nodes
       and values the communities
    graph : networkx.Graph
       the networkx graph which is decomposed
    weight : str, optional
        the key in graph to use as weight. Default to 'weight'


    Returns
    -------
    modularity : float
       The modularity

    Raises
    ------
    KeyError
       If the partition is not a partition of all graph nodes
    ValueError
        If the graph has no link
    TypeError
        If graph is not a networkx.Graph

    References
    ----------
    .. 1. Newman, M.E.J. & Girvan, M. Finding and evaluating community
    structure in networks. Physical Review E 69, 26113(2004).

    Examples
    --------
    >>> import community as community_louvain
    >>> import networkx as nx
    >>> G = nx.erdos_renyi_graph(100, 0.01)
    >>> partition = community_louvain.best_partition(G)
    >>> modularity(partition, G)
    """
    if graph.is_directed():
        raise TypeError("Bad graph type, use only non directed graph")

    inc = dict([])
    deg = dict([])
    links = graph.size(weight=weight)
    if links == 0:
        raise ValueError("A graph without link has an undefined modularity")

    for node in graph:
        com = partition[node]
        deg[com] = deg.get(com, 0.) + graph.degree(node, weight=weight)
        for neighbor, datas in graph[node].items():
            edge_weight = datas.get(weight, 1)
            if partition[neighbor] == com:
                if neighbor == node:
                    inc[com] = inc.get(com, 0.) + float(edge_weight)
                else:
                    inc[com] = inc.get(com, 0.) + float(edge_weight) / 2.

    res = 0.
    for com in set(partition.values()):
        res += (inc.get(com, 0.) / links) - \
               (deg.get(com, 0.) / (2. * links)) ** 2
    return res


def best_partition(graph,
                   partition=None,
                   weight='weight',
                   resolution=1.,
                   randomize=None,
                   random_state=None,
                   expected_matrix=None):
    """Compute the partition of the graph nodes which maximises the modularity
    (or try..) using the Louvain heuristices

    This is the partition of highest modularity, i.e. the highest partition
    of the dendrogram generated by the Louvain algorithm.

    Parameters
    ----------
    graph : networkx.Graph
       the networkx graph which is decomposed
    partition : dict, optional
       the algorithm will start using this partition of the nodes.
       It's a dictionary where keys are their nodes and values the communities
    weight : str, optional
        the key in graph to use as weight. Default to 'weight'
    resolution :  double, optional
        Will change the size of the communities, default to 1.
        represents the time described in
        "Laplacian Dynamics and Multiscale Modular Structure in Networks",
        R. Lambiotte, J.-C. Delvenne, M. Barahona
    randomize : boolean, optional
        Will randomize the node evaluation order and the community evaluation
        order to get different partitions at each call
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Returns
    -------
    partition : dictionnary
       The partition, with communities numbered from 0 to number of communities

    Raises
    ------
    NetworkXError
       If the graph is not undirected.

    See Also
    --------
    generate_dendrogram : to obtain all the decompositions levels

    Notes
    -----
    Uses Louvain algorithm

    References
    ----------
    .. 1. Blondel, V.D. et al. Fast unfolding of communities in
    large networks. J. Stat. Mech 10008, 1-12(2008).

    Examples
    --------
    >>> # basic usage
    >>> import community as community_louvain
    >>> import networkx as nx
    >>> G = nx.erdos_renyi_graph(100, 0.01)
    >>> partion = community_louvain.best_partition(G)

    >>> # display a graph with its communities:
    >>> # as Erdos-Renyi graphs don't have true community structure,
    >>> # instead load the karate club graph
    >>> import community as community_louvain
    >>> import matplotlib.cm as cm
    >>> import matplotlib.pyplot as plt
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> # compute the best partition
    >>> partition = community_louvain.best_partition(G)

    >>> # draw the graph
    >>> pos = nx.spring_layout(G)
    >>> # color the nodes according to their partition
    >>> cmap = cm.get_cmap('viridis', max(partition.values()) + 1)
    >>> nx.draw_networkx_nodes(G, pos, partition.keys(), node_size=40,
    >>>                        cmap=cmap, node_color=list(partition.values()))
    >>> nx.draw_networkx_edges(G, pos, alpha=0.5)
    >>> plt.show()
    """

    id_to_node = dict(enumerate(graph.nodes()))
    node_to_id = {v: k for k, v in id_to_node.items()}
    graph_new = nx.relabel_nodes(graph, node_to_id)

    if expected_matrix is not None:
        print("Using spatial modularity.")
    else:
        print("Using standard modularity.")

    dendo = generate_dendrogram(graph_new,
                                partition,
                                weight,
                                resolution,
                                randomize,
                                random_state,
                                expected_matrix)

    partition = partition_at_level(dendo, len(dendo) - 1)
    partition_original_labels = {id_to_node[node_id]: community for node_id, community in partition.items()}

    return partition_original_labels


def generate_dendrogram(graph,
                        part_init=None,
                        weight='weight',
                        resolution=1.,
                        randomize=None,
                        random_state=None,
                        expected_matrix=None):
    """Find communities in the graph and return the associated dendrogram

    A dendrogram is a tree and each level is a partition of the graph nodes.
    Level 0 is the first partition, which contains the smallest communities,
    and the best is len(dendrogram) - 1. The higher the level is, the bigger
    are the communities


    Parameters
    ----------
    graph : networkx.Graph
        the networkx graph which will be decomposed
    part_init : dict, optional
        the algorithm will start using this partition of the nodes. It's a
        dictionary where keys are their nodes and values the communities
    weight : str, optional
        the key in graph to use as weight. Default to 'weight'
    resolution :  double, optional
        Will change the size of the communities, default to 1.
        represents the time described in
        "Laplacian Dynamics and Multiscale Modular Structure in Networks",
        R. Lambiotte, J.-C. Delvenne, M. Barahona

    Returns
    -------
    dendrogram : list of dictionaries
        a list of partitions, ie dictionnaries where keys of the i+1 are the
        values of the i. and where keys of the first are the nodes of graph

    Raises
    ------
    TypeError
        If the graph is not a networkx.Graph

    See Also
    --------
    best_partition

    Notes
    -----
    Uses Louvain algorithm

    References
    ----------
    .. 1. Blondel, V.D. et al. Fast unfolding of communities in large
    networks. J. Stat. Mech 10008, 1-12(2008).

    Examples
    --------
    >>> G=nx.erdos_renyi_graph(100, 0.01)
    >>> dendo = generate_dendrogram(G)
    >>> for level in range(len(dendo) - 1) :
    >>>     print("partition at level", level,
    >>>           "is", partition_at_level(dendo, level))
    :param weight:
    :type weight:
    """
    if graph.is_directed():
        raise TypeError("Bad graph type, use only non directed graph")

    # Properly handle random state, eventually remove old `randomize` parameter
    # NOTE: when `randomize` is removed, delete code up to random_state = ...
    if randomize is not None:
        warnings.warn("The `randomize` parameter will be deprecated in future "
                      "versions. Use `random_state` instead.", DeprecationWarning)
        # If shouldn't randomize, we set a fixed seed to get determinisitc results
        if randomize is False:
            random_state = 0

    # We don't know what to do if both `randomize` and `random_state` are defined
    if randomize and random_state is not None:
        raise ValueError("`randomize` and `random_state` cannot be used at the "
                         "same time")

    random_state = check_random_state(random_state)

    # special case, when there is no link
    # the best partition is everyone in its community
    if graph.number_of_edges() == 0:
        part = dict([])
        for i, node in enumerate(graph.nodes()):
            part[node] = i
        return [part]

    use_spatial = expected_matrix is not None # using the spatial type if spatial weight matrix is given

    current_graph = graph.copy()
    status = Status()
    status.init(current_graph, weight, part=part_init, expected_matrix=(expected_matrix if use_spatial else None))
    status_list = list()
    __one_level(current_graph, status, weight, resolution, random_state, expected_matrix if use_spatial else None)
    new_mod = __modularity(status, resolution, expected_matrix if use_spatial else None)
    partition = __renumber(status.node2com)
    status_list.append(partition)
    mod = new_mod
    current_graph = induced_graph(partition, current_graph, weight) # aggregating the graph to super nodes and edges
    if expected_matrix is not None: # extending aggregation to the spatial weight matrix
        expected_matrix = induced_expected_matrix(partition, expected_matrix)
    status.init(current_graph, weight, part=partition, expected_matrix=(expected_matrix if use_spatial else None))

    while True:
        __one_level(current_graph, status, weight, resolution, random_state, expected_matrix if use_spatial else None)
        new_mod = __modularity(status, resolution, expected_matrix if use_spatial else None)
        if new_mod - mod < __MIN:
            break
        partition = __renumber(status.node2com)
        status_list.append(partition)
        mod = new_mod
        current_graph = induced_graph(partition, current_graph, weight)
        if expected_matrix is not None: # extending aggregation to the spatial weight matrix here as well
            expected_matrix = induced_expected_matrix(partition, expected_matrix) 
        status.init(current_graph, weight, part=partition, expected_matrix=(expected_matrix if use_spatial else None))
    return status_list[:]


def induced_expected_matrix(partition, expected_matrix):
    """Aggregate the expected matrix based on current partition (community → node)"""
    import numpy as np

    com_ids = list(set(partition.values()))
    com_index = {com: idx for idx, com in enumerate(com_ids)}
    n = len(com_ids)
    new_matrix = np.zeros((n, n))

    for i in range(len(expected_matrix)):
        for j in range(len(expected_matrix)):
            ci = partition[i]
            cj = partition[j]
            new_i = com_index[ci]
            new_j = com_index[cj]
            new_matrix[new_i, new_j] += expected_matrix[i, j]

    return new_matrix


def induced_graph(partition, graph, weight="weight"):
    """Produce the graph where nodes are the communities

    there is a link of weight w between communities if the sum of the weights
    of the links between their elements is w

    Parameters
    ----------
    partition : dict
       a dictionary where keys are graph nodes and  values the part the node
       belongs to
    graph : networkx.Graph
        the initial graph
    weight : str, optional
        the key in graph to use as weight. Default to 'weight'


    Returns
    -------
    g : networkx.Graph
       a networkx graph where nodes are the parts

    Examples
    --------
    >>> n = 5
    >>> g = nx.complete_graph(2*n)
    >>> part = dict([])
    >>> for node in g.nodes() :
    >>>     part[node] = node % 2
    >>> ind = induced_graph(part, g)
    >>> goal = nx.Graph()
    >>> goal.add_weighted_edges_from([(0,1,n*n),(0,0,n*(n-1)/2), (1, 1, n*(n-1)/2)])  # NOQA
    >>> nx.is_isomorphic(ind, goal)
    True
    """
    ret = nx.Graph()
    ret.add_nodes_from(partition.values())

    for node1, node2, datas in graph.edges(data=True):
        edge_weight = datas.get(weight, 1)
        com1 = partition[node1]
        com2 = partition[node2]
        w_prec = ret.get_edge_data(com1, com2, {weight: 0}).get(weight, 1)
        ret.add_edge(com1, com2, **{weight: w_prec + edge_weight})

    return ret


def __renumber(dictionary):
    """Renumber the values of the dictionary from 0 to n
    """
    values = set(dictionary.values())
    target = set(range(len(values)))

    if values == target:
        # no renumbering necessary
        ret = dictionary.copy()
    else:
        # add the values that won't be renumbered
        renumbering = dict(zip(target.intersection(values),
                               target.intersection(values)))
        # add the values that will be renumbered
        renumbering.update(dict(zip(values.difference(target),
                                    target.difference(values))))
        ret = {k: renumbering[v] for k, v in dictionary.items()}

    return ret


def load_binary(data):
    """Load binary graph as used by the cpp implementation of this algorithm
    """
    data = open(data, "rb")

    reader = array.array("I")
    reader.fromfile(data, 1)
    num_nodes = reader.pop()
    reader = array.array("I")
    reader.fromfile(data, num_nodes)
    cum_deg = reader.tolist()
    num_links = reader.pop()
    reader = array.array("I")
    reader.fromfile(data, num_links)
    links = reader.tolist()
    graph = nx.Graph()
    graph.add_nodes_from(range(num_nodes))
    prec_deg = 0

    for index in range(num_nodes):
        last_deg = cum_deg[index]
        neighbors = links[prec_deg:last_deg]
        graph.add_edges_from([(index, int(neigh)) for neigh in neighbors])
        prec_deg = last_deg

    return graph


def __one_level(graph, status, weight_key, resolution, random_state, expected_matrix=None):
    """Compute one level of communities
    """
    modified = True
    nb_pass_done = 0
    cur_mod = __modularity(status, resolution, expected_matrix)
    new_mod = cur_mod

    while modified and nb_pass_done != __PASS_MAX:
        cur_mod = new_mod
        modified = False
        nb_pass_done += 1

        for node in __randomize(graph.nodes(), random_state): # select a node randomly
            com_node = status.node2com[node] # get the community of the chosen node
            degc_totw = status.gdegrees.get(node, 0.) / (status.total_weight * 2.)  # NOQA  node degree divided by network degree
            neigh_communities = __neighcom(node, graph, status, weight_key) # this function computes the sum of edge weights to each neighboring communities, returning a dictionary to the particular node
            if expected_matrix is None:
                remove_cost = - neigh_communities.get(com_node,0) + \
                    resolution * (status.degrees.get(com_node, 0.) - status.gdegrees.get(node, 0.)) * degc_totw
            else:
                # in the spatial version, modularity loss comes from the negative difference between true and expected weight sum wich own communityy
                expected_contribution = 0.0
                for other_node in graph.nodes():
                    if status.node2com.get(other_node) == com_node:
                        if node == other_node:
                            expected_contribution += expected_matrix[node, other_node]
                        else:
                            expected_contribution += expected_matrix[node, other_node] / 2.0
                remove_cost = - neigh_communities.get(com_node, 0) + expected_contribution * resolution # do we need resolution?
            __remove(node, com_node,
                     neigh_communities.get(com_node, 0.), status, graph, expected_matrix)
            best_com = com_node
            best_increase = 0
            for com, dnc in __randomize(neigh_communities.items(), random_state):
                if expected_matrix is None:
                    incr = remove_cost + dnc - status.degrees.get(com, 0.) * degc_totw * resolution # do we need resolution?
                else:
                    # Compute expected weight between node and target community
                    expected_contribution = 0.0
                    for other_node in graph.nodes():
                        if status.node2com.get(other_node) == com:
                            if node == other_node:
                                expected_contribution += expected_matrix[node, other_node]
                            else:
                                expected_contribution += expected_matrix[node, other_node] / 2.0
                    incr = dnc - expected_contribution  # spatial modularity gain
                if incr > best_increase:
                    best_increase = incr
                    best_com = com
            __insert(node, best_com,
                     neigh_communities.get(best_com, 0.), status, graph, expected_matrix)
            if best_com != com_node:
                modified = True
        new_mod = __modularity(status, resolution, expected_matrix)
        if new_mod - cur_mod < __MIN:
            break


def __neighcom(node, graph, status, weight_key): # this function computes the sum of edge weights to each neighboring communities, returning a dictionary to the particular node
    """
    Compute the communities in the neighborhood of node in the graph given
    with the decomposition node2com
    """
    weights = {}
    for neighbor, datas in graph[node].items():
        if neighbor != node:
            edge_weight = datas.get(weight_key, 1)
            neighborcom = status.node2com[neighbor]
            weights[neighborcom] = weights.get(neighborcom, 0) + edge_weight

    return weights


def __remove(node, com, weight, status, graph, expected_matrix=None):
    """ Remove node from community com and modify status"""
    status.degrees[com] = (status.degrees.get(com, 0.)
                           - status.gdegrees.get(node, 0.))
    status.internals[com] = float(status.internals.get(com, 0.) -
                                  weight - status.loops.get(node, 0.))
    status.node2com[node] = -1

    if expected_matrix is not None: # updating the internal weight sum of expected edges as well
        delta = 0.0
        for other_node in graph.nodes():
            if status.node2com[other_node] == com:
                if other_node == node:
                    delta += expected_matrix[node, node]
                else:
                    delta += expected_matrix[node, other_node] / 2.0
        status.expected_internal[com] -= delta


def __insert(node, com, weight, status, graph, expected_matrix=None):
    """ Insert node into community and modify status"""
    status.node2com[node] = com
    status.degrees[com] = (status.degrees.get(com, 0.) +
                           status.gdegrees.get(node, 0.))
    status.internals[com] = float(status.internals.get(com, 0.) +
                                  weight + status.loops.get(node, 0.))
    
    if expected_matrix is not None: # similarly updating the internal weight sum of expected edges as well
        delta = 0.0
        for other_node in graph.nodes():
            if status.node2com[other_node] == com:
                if other_node == node:
                    delta += expected_matrix[node, node]
                else:
                    delta += expected_matrix[node, other_node] / 2.0
        status.expected_internal[com] = status.expected_internal.get(com, 0.0) + delta


def __modularity(status, resolution, expected_matrix=None):
    links = float(status.total_weight)
    if expected_matrix is None:
        result = 0.
        for community in set(status.node2com.values()):
            in_degree = status.internals.get(community, 0.)
            degree = status.degrees.get(community, 0.)
            if links > 0:
                result += in_degree * resolution / links - ((degree / (2. * links)) ** 2)
        return result
    else:
        expected_links = float(status.expected_total_weight)
        result = 0.
        for community in set(status.node2com.values()):
            internal_weight = status.internals.get(community, 0.)
            expected_weight = status.expected_internal.get(community, 0.)
            if links > 0:
                result += internal_weight * resolution / links - expected_weight / links
        return result


def __randomize(items, random_state):
    """Returns a List containing a random permutation of items"""
    randomized_items = list(items)
    random_state.shuffle(randomized_items)
    return randomized_items
