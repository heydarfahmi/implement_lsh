import json
import numbers
import pickle
import random
import time
from collections import defaultdict
from itertools import chain

import numpy as np
from scipy.sparse import csr_matrix, issparse

from src.LSH.lsh import LSH


def jaccard_dissimilarity(a, b, **_):
    return np.array([sum(b != aa) for aa in a])


def cosine_dissimilarity():
    pass


def get_max_value_key(dic):
    """Gets the key for the maximum value in a dict."""
    v = np.array(list(dic.values()))
    k = np.array(list(dic.keys()))

    maxima = np.where(v == np.max(v))[0]
    if len(maxima) == 1:
        return k[maxima[0]]
    # In order to be consistent, always selects the minimum key
    # (guaranteed to be unique) when there are multiple maximum values.
    return k[maxima[np.argmin(k[maxima])]]


def get_unique_rows(a):
    """Gets the unique rows in a numpy array."""
    return np.vstack(list({tuple(row) for row in a}))


def get_random_state(seed=None):
    """
    Get a random number generator (RandomState or Generator) based on the provided seed.

    Parameters:
    - seed: None, int, numpy.random.Generator, or numpy.random.RandomState
        The seed or generator to use for random number generation.

    Returns:
    - numpy.random.Generator or numpy.random.RandomState
        A random number generator instance.

    Raises:
    - ValueError
        If the seed is not of a valid type.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, numbers.Integral):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError(
        "%r cannot be used to seed a numpy.random.RandomState instance" % seed
    )


def _update_centroids(X, n_clusters, centroids_number, labels):
    centroids = [None for _ in range(n_clusters)]
    for ipoint, point in enumerate(X):
        label = labels[ipoint]
        if centroids[label] is None:
            centroids[label] = point
        else:
            centroids[label] += point
    for k in range(n_clusters):
        centroids[k] = centroids[k].toarray()[0] / centroids_number[k]

    return np.array(centroids)


def perform_empty_clusters(X, n_clusters, random_state,
                           centroids, membership_matrix, centroids_number, labels):
    moves = 0
    for ik in range(n_clusters):
        if centroids_number[ik] == 0:
            from_clust = centroids_number.argmax()
            choices = [ii for ii, ch in enumerate(membership_matrix[from_clust, :]) if ch]
            rindx = random_state.choice(choices)
            membership_matrix[from_clust, rindx] = 0
            membership_matrix[ik, rindx] = 1
            centroids_number[from_clust] -= 1
            centroids_number[ik] += 1
            labels[rindx] = ik
            centroids[ik] = X[rindx].toarray()[0]

            moves += 1

    return labels, centroids_number, membership_matrix, centroids, moves


def _assign_clusters(X, centroids, membership_matrix, centroids_number, fdissimilarity, labels):
    moves = 0
    costs = 0
    for ipoint, curpoint in enumerate(X):
        diss = fdissimilarity(centroids, curpoint)
        clust = np.argmin(diss)
        costs += diss[clust]

        if membership_matrix[clust, ipoint]:
            continue
        moves += 1

        membership_matrix[labels[ipoint], ipoint] = 0
        membership_matrix[clust, ipoint] = 1

        centroids_number[labels[ipoint]] -= 1
        centroids_number[clust] += 1

        labels[ipoint] = clust

    return labels, centroids_number, membership_matrix, moves, costs


def _assign_bucket_clusters(X, lsh_index, bucket_clusters, centroids, membership_matrix, centroids_number,
                            fdissimilarity, labels):
    moves = 0
    costs = 0
    for ipoint, curpoint in enumerate(X):
        short_list = []
        for bucket_index, bucket in enumerate(lsh_index.keys[ipoint]):
            short_list.append(bucket_clusters[bucket_index][bucket])

        to_diss = np.array(list(chain(*short_list)))
        diss = fdissimilarity(centroids[to_diss], curpoint)
        _clust = np.argmin(diss)
        costs += diss[_clust]
        clust = to_diss[_clust]

        if membership_matrix[clust, ipoint]:
            continue
        moves += 1

        membership_matrix[labels[ipoint], ipoint] = 0
        membership_matrix[clust, ipoint] = 1

        centroids_number[labels[ipoint]] -= 1
        centroids_number[clust] += 1

        labels[ipoint] = clust

    return labels, centroids_number, membership_matrix, moves, costs


def query_short_list(X, lsh_index: LSH, labels):
    bucket_short_list = [defaultdict(set) for _ in range(lsh_index.b)]
    for ipoint, point in enumerate(X):
        for bucket_index, hash in enumerate(lsh_index.keys[ipoint]):
            bucket_short_list[bucket_index][hash].add(labels[ipoint])

    return bucket_short_list


def _mhkmeans(X, lsh_index, n_clusters, n_points, n_attrs, max_iter, fdissimilarity, init, init_no,
              verbose, random_state, save, save_dir='.'):
    random_state = get_random_state(random_state)
    # _____ INIT STEP_____
    if verbose:
        print("Init: initializing centroids")
    if isinstance(init, str) and init.lower() == 'lsh++':
        pass
    elif isinstance(init, str) and init.lower() == 'kmeans++':
        pass
    elif isinstance(init, str) and init.lower() == 'random':
        sample_weight = np.ones(n_points, dtype=X.dtype)
        seeds = random_state.choice(
            n_points,
            size=n_clusters,
            replace=False,
            p=sample_weight / sample_weight.sum(),
        )
        centroids = X[seeds]
    elif hasattr(init, '__array__'):
        # Make sure init is a 2D array.
        if len(init.shape) == 1:
            init = np.atleast_2d(init).T
        assert init.shape[0] == n_clusters, \
            f"Wrong number of initial centroids in init ({init.shape[0]}, " \
            f"should be {n_clusters})."
        assert init.shape[1] == n_attrs, \
            f"Wrong number of attributes in init ({init.shape[1]}, " \
            f"should be {n_attrs})."
        centroids = np.asarray(init, dtype=np.uint16)
    else:
        raise NotImplementedError

    if issparse(X):
        centroids = centroids.toarray()

    else:
        # todo convert X to sparse array
        pass

    if verbose:
        print("Init: initializing clusters")

    start_time = time.time()
    membership_matrix = np.zeros((n_clusters, n_points), dtype=np.bool_)

    labels = np.empty(n_points, dtype=np.uint16)
    centroids_number = np.array([0 for _ in range(n_clusters)])
    for ipoint, curpoint in enumerate(X):
        # Initial assignment to clusters
        diss = fdissimilarity(centroids, curpoint)
        clust = np.argmin(diss)
        membership_matrix[clust, ipoint] = 1
        centroids_number[clust] += 1
        labels[ipoint] = clust

    labels, centroids_number, membership_matrix, centroids, moves = perform_empty_clusters(X, n_clusters, random_state,
                                                                                           centroids,
                                                                                           membership_matrix,
                                                                                           centroids_number, labels)

    centroids = _update_centroids(X, n_clusters, centroids_number, labels)
    labels, centroids_number, membership_matrix, moves, cost = _assign_clusters(X, centroids, membership_matrix,
                                                                                centroids_number, fdissimilarity,
                                                                                labels)
    end_time = time.time()

    # _____ ITERATION _____
    if verbose:
        print("Starting iterations...")

    if save == 'data':
        rdata = {'step': 'initalization', 'init_no': init_no + 1, 'cost': cost, 'moves': 0,
                 'centers': centroids, labels: 'labels', 'step_time': end_time - start_time}
        with open(f'{save_dir}/{init_no}_initalization.pkl', 'wb') as f:
            pickle.dump(rdata, f)

    if save == 'timing' or save == 'data':
        rtime = {'step': 'initalization', 'init_no': init_no + 1, 'cost': float(cost), 'moves': 0,
                 'step_time': end_time - start_time}
        with open(f'{save_dir}/{init_no}_initalization.json', 'w') as f:
            f.write(json.dumps(rtime))

    itr = 0
    converged = False
    scape_pulse = False
    epoch_costs = [cost]
    while itr < max_iter and not converged:
        itr += 1
        iter_time_start = time.time()
        moves = 0
        labels, centroids_number, membership_matrix, centroids, _moves = perform_empty_clusters(X, n_clusters,
                                                                                                random_state,
                                                                                                centroids,
                                                                                                membership_matrix,
                                                                                                centroids_number,
                                                                                                labels)
        perform_time_step = time.time()
        moves += _moves
        centroids = _update_centroids(X, n_clusters, centroids_number, labels)

        update_time_step = time.time()
        if not scape_pulse:
            bucket_clusters = query_short_list(X, lsh_index, labels)
            labels, centroids_number, membership_matrix, _moves, ncost = _assign_bucket_clusters(X, lsh_index,
                                                                                                 bucket_clusters,
                                                                                                 centroids,
                                                                                                 membership_matrix,
                                                                                                 centroids_number,
                                                                                                 fdissimilarity,
                                                                                                 labels)
        else:
            labels, centroids_number, membership_matrix, _moves, ncost = _assign_clusters(X,
                                                                                          centroids,
                                                                                          membership_matrix,
                                                                                          centroids_number,
                                                                                          fdissimilarity,
                                                                                          labels)

        moves += _moves
        iter_time_end = time.time()

        # All points seen in this iteration
        _converged = (moves == 0) or (ncost >= cost)
        # TEST
        converged = scape_pulse and _converged
        scape_pulse = _converged

        epoch_costs.append(ncost)
        cost = ncost
        if verbose:
            print(f"Run {init_no + 1}, iteration: {itr}/{max_iter}, "
                  f"moves: {moves}, cost: {cost} in {iter_time_end - iter_time_start} seconds")

        if save == 'data':
            rdata = {'step': itr, 'init_no': init_no + 1, 'cost': cost, 'moves': 0,
                     'centers': centroids, labels: 'labels', 'step_time': iter_time_end - iter_time_start}
            with open(f'{save_dir}/{init_no}_{itr}.pkl', 'wb') as f:
                pickle.dump(rdata, f)

        if save == 'timing' or save == 'data':
            rtime = {'step': itr, 'init_no': init_no + 1, 'cost': float(cost), 'moves': 0,
                     'assignment_time': iter_time_end - update_time_step,
                     'perform_step_time': perform_time_step - iter_time_start,
                     'update_centroid_time': update_time_step - perform_time_step,
                     'step_time': iter_time_end - iter_time_start}

            with open(f'{save_dir}/{init_no}_{itr}.json', 'w') as f:
                f.write(json.dumps(rtime))

    return centroids, labels, cost, itr, epoch_costs


def mhkmeans(X, lsh_index, n_clusters, max_iter, fdissimilarity, init, n_init, verbose, random_state, save, save_dir
             ):
    """k-modes algorithm"""
    random_state = get_random_state(random_state)

    # Todo assert if X has true instances

    n_points, n_attrs = X.shape
    assert n_clusters <= n_points, f"Cannot have more clusters ({n_clusters}) " \
                                   f"than data points ({n_points})."

    # Are there more n_clusters than unique rows? Then set the unique
    # rows as initial values and skip iteration.
    unique = get_unique_rows(X)
    n_unique = unique.shape[0]
    if n_unique <= n_clusters:
        max_iter = 0
        n_init = 1
        n_clusters = n_unique
        init = unique

    results = []
    seeds = random_state.randint(np.iinfo(np.int32).max, size=n_init)
    for init_no in range(n_init):
        results.append(_mhkmeans(
            X, lsh_index, n_clusters, n_points, n_attrs, max_iter, fdissimilarity, init, init_no,
            verbose, seeds[init_no], save, save_dir
        ))

    all_centroids, all_labels, all_costs, all_n_iters, all_epoch_costs = zip(*results)

    best = np.argmin(all_costs)
    if n_init > 1 and verbose:
        print(f"Best run was number {best + 1}")

    return all_centroids[best], all_labels[best], \
        all_costs[best], all_n_iters[best], all_epoch_costs[best]


class Kmeans:

    def __init__(self, n_clusters=8, max_iteration=100, dissimilarity='jaccard',
                 init='random', random_state=None, n_init=10, **kwargs):
        self.n_clusters = n_clusters
        self.max_iter = max_iteration
        self.dissimilarity = dissimilarity
        self.init = init
        self.random_state = random_state
        self.verbose = kwargs.get('verbose', False)
        self.save = kwargs.get('save')
        self.save_dir = kwargs.get('save_dir')

    def fit(self, X, **kwargs):
        """Compute k-means clustering.

          Parameters
          ----------
          X : array-like, shape=[n_samples, n_features]

          The weight that is assigned to each individual data point when
          updating the centroids.
          """

        # TODO check if X i numpy

        random_state = get_random_state(self.random_state)

        (self._enc_cluster_centroids, self._enc_map, self.labels_, self.cost_,
         self.n_iter_, self.epoch_costs_) = mhkmeans(
            X,
            self.n_clusters,
            self.max_iter,
            self.dissimilarity,
            self.init,
            self.n_init,
            self.verbose,
            random_state,
            self.save,
            self.save_dir)
        return self

    def fit_predict(self, X, y=None, **kwargs):
        """Compute cluster centroids and predict cluster index for each sample.

        Convenience method; equivalent to calling fit(X) followed by
        predict(X).
        """
        return self.fit(X, **kwargs).predict(X, **kwargs)

    def predict(self, X, **kwargs):
        assert hasattr(self, '_enc_cluster_centroids'), "Model not yet fitted."

        # todo check array
        return labels_cost(X, self._enc_cluster_centroids, self.cat_dissim)[0]

    @property
    def cluster_centroids_(self):
        pass
