"""
graph obs implementation computer.
"""
from abc import ABC

import jax.numpy as np
import numpy as onp
import jax
import jax.numpy as np
from jax import random
import numpy as onp
import jraph
import jax.tree_util as tree
import networkx as nx
import optax
import functools
import haiku as hk
from jraph._src import graph as gn_graph
from jraph._src import utils
import flax.linen as nn
from flax.training.train_state import TrainState
from optax._src.base import GradientTransformation
from typing import Any, Callable, Dict, List, Optional, Tuple


from .observable import Observable


def _angle_and_dist(colloid, colloids):
    # angles between the colloid director and line of sight to other colloids
    angles = []
    # angles between colloid director and director of other colloids
    angles2 = []
    dists = []
    my_director = colloid.director[:2]
    for col in colloids:
        if col is not colloid:
            my_col_vec = col.pos[:2] - colloid.pos[:2]
            my_col_dist = np.linalg.norm(my_col_vec)

            # compute angle 1
            my_col_vec = my_col_vec / my_col_dist
            angle = np.arccos(np.dot(my_col_vec, my_director))
            orthogonal_dot = np.dot(my_col_vec,
                                    np.array([-my_director[1], my_director[0]]))
            angle *= np.sign(orthogonal_dot) / np.pi

            # compute angle 2
            other_director = col.director[:2]
            angle2 = np.arccos(np.dot(other_director, my_director))
            orthogonal_dot2 = np.dot(other_director,
                                     np.array([-my_director[1], my_director[0]]))
            angle2 *= np.sign(orthogonal_dot2) / np.pi
            angles2.append(angle2)
            angles.append(angle)
            dists.append(my_col_dist)
    return np.array(angles), np.array(angles2), np.array(dists)


class GraphObs(Observable, ABC):
    """
    Implementation of the GraphOps observable.
    """

    def __init__(self,
                 box_size,
                 r_cut: float,
                 encoder_network: nn.Module,
                 node_updater_network: nn.Module,
                 influencer_network: nn.Module,
                 obs_shape: int = 8,
                 relate=False,
                 attention_normalize_fn=utils.segment_softmax,
                 seed=42,
                 ):

        self.box_size = box_size
        self.r_cut = r_cut
        self.obs_shape = obs_shape
        self.relate = relate
        self.attention_normalize_fn = attention_normalize_fn

        # ML part of the observable
        self.rngkey = random.PRNGKey(seed)
        self.networks = {"encoder": encoder_network,
                         "node_updater": node_updater_network,
                         "influencer": influencer_network}
        self.states = {"encoder": None,
                       "node_updater": None,
                       "influencer": None}

        self.encode_fn, self.update_node_fn, self.influence_eval_fn = self._init_models()

    def initialize(self, colloids: list):
        pass

    def _init_models(self):
        split, self.rngkey = random.split(key=self.rngkey)
        for key, item in self.networks.items():
            rngkey, split = random.split(key=split)
            params = item.init(rngkey, np.ones(4))["params"]
            self.states[key] = TrainState.create(apply_fn=jax.jit(item.apply),
                                                 params=params,
                                                 tx=optax.adam(learning_rate=0.001)
        )

        encoder = jax.jit(self.networks["encoder"].apply)

        def encode_fn(features):
            encoded = encoder({"params": self.states["encoder"].params}, features)
            return encoded

        node_updater = jax.jit(self.networks["node_updater"].apply)

        def node_update_fn(features):
            updated = node_updater({"params": self.states["node_updater"].params},
                                   features)
            return updated

        influencer = jax.jit(self.networks["influencer"].apply)

        def influencer_fn(features):
            influenced = influencer({"params": self.states["influencer"].params},
                                    features)
            return influenced

        return encode_fn, node_update_fn, influencer_fn

    def _update_models(self,
                       grads,
                       ):
        # questionable how the grads will come back! Let's see at gradient computation
        for key, item in self.states:
            self.states[key] = self.states[key].apply_gradients(grads=grads)

    def _build_graph(self, colloid, colloids):
        nodes = []
        angles, angles2, dists = _angle_and_dist(colloid, colloids)
        node_index = 0
        for i, col in enumerate(colloids):
            if col is not colloid:
                r = dists[i]
                if r < self.r_cut:
                    node_index += 1
                    node = np.hstack(((dists[i] / self.box_size),
                                      angles[i],
                                      angles2[i],
                                      col.type))
                    nodes.append(node)
        graph = utils.get_fully_connected_graph(n_node_per_graph=len(nodes),
                                                n_graph=1,
                                                node_features=np.array(nodes),
                                                add_self_edges=False)
        return graph

    def compute_observable(self,
                           colloid: object,
                           other_colloids: list,
                           return_graph= False):

        graph = self._build_graph(colloid, other_colloids)

        nodes, edges, receivers, senders, globals_, n_node, n_edge = graph

        # encode node features n_i to attention_vec a_i
        attention_vectors = tree.tree_map(lambda n: self.encode_fn(n), nodes)

        if self.relate:

            # function 1
            sent_attributes, sent_attention = tree.tree_map(
                lambda n, a: (n[senders], a[senders]),
                nodes,
                attention_vectors)
            # function 2
            received_attributes, received_attention = tree.tree_map(
                lambda n, a: (n[receivers], a[receivers]),
                nodes,
                attention_vectors)
            # function 3
            # this can be made to a learnable matrix norm.
            edges = tree.tree_map(
                lambda r, s: np.exp(-np.linalg.norm(r - s, axis=1) ** 2),
                received_attention,
                sent_attention)
            # function 4
            # softmax
            tree_calculate_weights = functools.partial(utils.segment_softmax,
                                                       segment_ids=receivers,
                                                       num_segments=n_node
                                                       )
            weights = tree.tree_map(tree_calculate_weights, edges)

            # function 5
            received_weighted_attributes = tree.tree_map(
                lambda r, w: r * w[:, None],
                nodes[receivers],
                weights)
            # function 6
            received_message = utils.segment_sum(received_weighted_attributes,
                                                 receivers,
                                                 num_segments=n_node)
            # function 7
            nodes = self.update_node_fn(received_message)

            influence_score = self.influence_eval_fn(attention_vectors)
        else:
            influence_score = self.influence_eval_fn(nodes)

        influence = jax.nn.softmax(influence_score)

        # computes the actual feature
        graph_representation = np.sum(tree.tree_map(lambda n, i: n * i,
                                                    attention_vectors,
                                                    influence),
                                      axis=0
                                      )

        if not return_graph:

            return graph_representation

        else:
            return gn_graph.GraphsTuple(
                nodes=nodes,
                edges=edges,
                receivers=receivers,
                senders=senders,
                globals=graph_representation,
                n_node=n_node,
                n_edge=n_edge)

