from collections import defaultdict
from scipy import sparse
import numpy as np
import numbers
import time
import pickle
import json
from src.benchmark.local import Kmodes


def jaccard_dissimilarity(a: set, b: set, **_):
    bsize = len(b)
    return np.array([len(aa) + bsize - 2*len(b.intersection(aa)) for aa in a])


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
        return np.random.default_rng()
    if isinstance(seed, numbers.Integral):
        return np.random.default_rng(seed)
    if isinstance(seed, np.random.Generator):
        return seed
    raise ValueError(f"{seed} cannot be used to seed a numpy.random.Generator instance")


def _update_centroids(X, centroid_N, n_clusters, labels):
    _cluster_attr_fq = [defaultdict(int)
                        for _ in range(n_clusters)]
    for ipoint, point in enumerate(X):
        for attr_value in point:
            cl = labels[ipoint]
            _cluster_attr_fq[cl][attr_value] += 1
    centroids = [{attr
                  for attr, attr_freq in _cluster_attr_fq[ipoint].items()
                  if attr_freq > centroid_N[ipoint]/2}
                 for ipoint in range(n_clusters)]
    return centroids, _cluster_attr_fq


def perform_empty_clusters(X, n_clusters, random_state,
                           centroids, membership_matrix, labels, fdissimilarity):
    moves = 0
    costs = 0
    for ik in range(n_clusters):
        if sum(membership_matrix[ik]) == 0:
            from_clust = membership_matrix.sum(axis=1).argmax()
            choices = [ii for ii, ch in enumerate(membership_matrix[from_clust, :]) if ch]
            rindx = random_state.choice(choices)
            membership_matrix[from_clust, rindx] = 0
            membership_matrix[ik, rindx] = 1
            labels[rindx] = ik
            centroids[ik] = X[rindx]

            moves += 1
            # TODO
            costs += fdissimilarity(np.array([centroids[from_clust]]), X[rindx], X=X, type='p2p')[0]
    centroids_num = membership_matrix.sum(axis=1)
    return labels, membership_matrix, centroids, centroids_num, moves, costs


def _assign_clusters(X, centroids, membership_matrix, fdissimilarity, labels):
    cost = 0.
    moves = 0
    for ipoint, curpoint in enumerate(X):
        diss = fdissimilarity(centroids, curpoint, X=X, membship=membership_matrix)
        clust = np.argmin(diss)
        cost += diss[clust]

        if membership_matrix[clust, ipoint]:
            continue

        moves += 1
        membership_matrix[labels[ipoint], ipoint] = 0
        labels[ipoint] = clust
        membership_matrix[clust, ipoint] = 1
    return labels, membership_matrix, moves, cost


def _bkmodes(X, n_clusters, n_points, n_attrs, max_iter, fdissimilarity, init, init_no,
             verbose, random_state, save, save_dir='.'):
    random_state = get_random_state(random_state)
    # _____ INIT STEP_____
    if verbose:
        print("Init: initializing centroids")
    if isinstance(init, str) and init.lower() == 'lsh++':
        pass
    elif isinstance(init, str) and init.lower() == 'random':
        seeds = random_state.choice(range(n_points), n_clusters)
        centroids = X[seeds]
    elif hasattr(init, '__array__'):
        # Make sure init is a 2D array.
        if len(init.shape) == 1:
            init = np.atleast_2d(init).T
        assert init.shape[0] == n_clusters, \
            f"Wrong number of initial centroids in init ({init.shape[0]}, " \
            f"should be {n_clusters})."
        # assert init.shape[1] == n_attrs, \
        #     f"Wrong number of attributes in init ({init.shape[1]}, " \
        #     f"should be {n_attrs})."
        #todo check
        centroids = init
    else:
        # todo check
        centroids = init
        # raise NotImplementedError

    if verbose:
        print("Init: initializing clusters")

    start_time = time.time()
    membership_matrix = np.zeros((n_clusters, n_points), dtype=np.bool_)

    centroids_num = [0 for _ in range(n_clusters)]
    labels = np.empty(n_points, dtype=np.uint16)
    cost = 0

    for ipoint, curpoint in enumerate(X):
        # Initial assignment to clusters
        diss = fdissimilarity(centroids, curpoint, X=X, membship=membership_matrix)
        clust = np.argmin(diss)
        cost += diss[clust]
        centroids_num[clust] += 1
        membership_matrix[clust, ipoint] = 1
        labels[ipoint] = clust
    centroids, _cluster_attrs = _update_centroids(X, centroids_num, n_clusters, labels)
    # _moves, _ncost = perform_empty_clusters(X, n_clusters, random_state, centroids, membership_matrix,
    #                                         fdissimilarity)

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

    epoch_costs = [cost]
    while itr < max_iter and not converged:
        itr += 1
        iter_time_start = time.time()
        labels, membership_matrix, moves, ncost = _assign_clusters(X, centroids, membership_matrix, fdissimilarity,
                                                                   labels)
        assing_time_step = time.time()

        labels, membership_matrix, centroids, centroids_num, _moves, _ncost = perform_empty_clusters(X, n_clusters,
                                                                                                     random_state,
                                                                                                     centroids,
                                                                                                     membership_matrix,
                                                                                                     labels,
                                                                                                     fdissimilarity)
        perform_time_step = time.time()
        moves += moves
        ncost -= _ncost
        centroids, _cluster_attr = _update_centroids(X, centroids_num, n_clusters, labels)

        iter_time_end = time.time()

        # All points seen in this iteration
        converged = (moves == 0) or (ncost >= cost)
        epoch_costs.append(ncost)
        cost = ncost
        if verbose:
            print(f"Run {init_no + 1}, iteration: {itr}/{max_iter}, "
                  f"moves: {moves}, cost: {cost}")

        if save == 'data':
            rdata = {'step': itr, 'init_no': init_no + 1, 'cost': cost, 'moves': 0,
                     'centers': centroids, labels: 'labels', 'step_time': iter_time_end - iter_time_start}
            with open(f'{save_dir}/{init_no}_{itr}.pkl', 'wb') as f:
                pickle.dump(rdata, f)

        if save == 'timing' or save == 'data':
            rtime = {'step': itr, 'init_no': init_no + 1, 'cost': float(cost), 'moves': 0,
                     'assignment_time': assing_time_step - iter_time_start,
                     'perform_step_time': perform_time_step - assing_time_step,
                     'update_centroid_time': iter_time_end - perform_time_step,
                     'step_time': iter_time_end - iter_time_start}
            with open(f'{save_dir}/{init_no}_{itr}.json', 'w') as f:
                f.write(json.dumps(rtime))

    return centroids, labels, cost, itr, epoch_costs


def bk_modes(X, n_clusters, max_iter, fdissimilarity, init, n_init, verbose, random_state, save, save_dir
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
        results.append(_bkmodes(
            X, n_clusters, n_points, n_attrs, max_iter, fdissimilarity, init, init_no,
            verbose, seeds[init_no], save, save_dir
        ))

    all_centroids, all_labels, all_costs, all_n_iters, all_epoch_costs = zip(*results)

    best = np.argmin(all_costs)
    if n_init > 1 and verbose:
        print(f"Best run was number {best + 1}")

    return all_centroids[best], all_labels[best], \
        all_costs[best], all_n_iters[best], all_epoch_costs[best]


class Kmodes:

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
        """Compute k-modes clustering.

          Parameters
          ----------
          X : array-like, shape=[n_samples, n_features]

          The weight that is assigned to each individual data point when
          updating the centroids.
          """

        # TODO check if X i numpy

        random_state = get_random_state(self.random_state)

        (self._enc_cluster_centroids, self._enc_map, self.labels_, self.cost_,
         self.n_iter_, self.epoch_costs_) = bk_modes(
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
        pass

    @property
    def cluster_centroids_(self):
        pass


if __name__ == '__main__':
    data2 = [{0, 2, 3},
             {1, 2, 3},
             {0, 1},
             {0, 1, 3},
             {0, 1}]
    modes2 = [
        {0, 2, 3},
        {2, 3}
    ]
    n_points, n_attrs = 5, 4
    _bkmodes(data2, 2, n_points, n_attrs, 10,
             jaccard_dissimilarity, modes2, 2, True, None,
             'timing', '/home/heydar/me/BSC/FinalPorject/lsh/src/benchmark/local/Result/t2')
