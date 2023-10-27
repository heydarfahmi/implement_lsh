import json
import numbers
import pickle
import random
import time
from collections import defaultdict
from itertools import chain

import numpy as np
from scipy.sparse import csr_matrix, issparse
from sklearn.utils.validation import check_array

from src.Kmeans.kmeans import Kmeans
from src.LSH.lsh import LSH


def _is_arraylike(x):
    """Returns whether the input is array-like and not a scalar."""
    return (hasattr(x, "__len__") or hasattr(x, "shape") or hasattr(x, "__array__")) and not np.isscalar(x)


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

        to_diss=list(chain(*short_list))
        to_diss = np.array(to_diss) if len(to_diss)>0 else np.array([labels[ipoint]])
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


def _mhkmeans(X, lsh_index, n_clusters, n_points, max_iter, fdissimilarity, centroids, init_no,
              verbose, random_state, save, save_dir='.'):
    random_state = get_random_state(random_state)
    # _____ INIT STEP_____
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


class MHKmeans(Kmeans):
    """MinHash K-Means clustering.

    Read more in the :ref:`User Guide <k_means>`.

    Parameters
    ----------

    n_clusters : int, default=8
        The number of clusters to form as well as the number of
        centroids to generate.

    init : {'k-means++', 'random'}, callable or array-like of shape \
            (n_clusters, n_features), default='k-means++'
        Method for initialization:

        'k-means++' : selects initial cluster centroids using sampling based on
        an empirical probability distribution of the points' contribution to the
        overall inertia. This technique speeds up convergence. The algorithm
        implemented is "greedy k-means++". It differs from the vanilla k-means++
        by making several trials at each sampling step and choosing the best centroid
        among them.

        'random': choose `n_clusters` observations (rows) at random from data
        for the initial centroids.

        If an array is passed, it should be of shape (n_clusters, n_features)
        and gives the initial centers.


    n_init :  int, default=10
        Number of times the k-means algorithm is run with different centroid
        seeds. The final results is the best output of `n_init` consecutive runs
        in terms of inertia. Several runs are recommended for sparse
        high-dimensional problems (see :ref:`kmeans_sparse_high_dim`).

        n_init could change depends on the value of init:
        5 if using `init='random'` ;
        1 if using `init='k-means++'` or `init` is an array-like.


    max_iter : int, default=100
        Maximum number of iterations of the k-means algorithm for a
        single run.


    verbose : int, default=0
        Verbosity mode.

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for centroid initialization. Use
        an int to make the randomness deterministic.
        See :term:`Glossary <random_state>`.


    Attributes
    ----------
    cluster_centers_ : ndarray of shape (n_clusters, n_features)
        Coordinates of cluster centers. If the algorithm stops before fully
        converging (see ``tol`` and ``max_iter``), these will not be
        consistent with ``labels_``.

    labels_ : ndarray of shape (n_samples,)
        Labels of each point

    inertia_ : float
        Sum of squared distances of samples to their closest cluster center,
        weighted by the sample weights if provided.

    cost_ : float
        equal to interia_.

    n_iter_ : int
        Number of iterations run.




    In practice, the k-means algorithm is very fast (one of the fastest
    clustering algorithms available), but it falls in local minima. That's why
    it can be useful to restart it several times.

    If the algorithm stops before fully converging (because of ``tol`` or
    ``max_iter``), ``labels_`` and ``cluster_centers_`` will not be consistent,
    i.e. the ``cluster_centers_`` will not be the means of the points in each
    cluster. Also, the estimator will reassign ``labels_`` after the last
    iteration to make ``labels_`` consistent with ``predict`` on the training
    set.

    Examples
    --------

    >>> from Kmeans.kmeans import KMeans
    >>> import numpy as np
    >>> X = np.array([[1, 2], [1, 4], [1, 0],
    ...               [10, 2], [10, 4], [10, 0]])
    >>> kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(X)
    >>> kmeans.labels_
    array([1, 1, 1, 0, 0, 0], dtype=int32)
    >>> kmeans.predict([[0, 0], [12, 3]])
    array([1, 0], dtype=int32)
    >>> kmeans.cluster_centers_
    array([[10.,  2.],
           [ 1.,  2.]])
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fit(self, X, lsh, **kwargs):
        """Compute k-means clustering.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training instances to cluster. It must be noted that the data
            will be converted to C ordering, which will cause a memory
            copy if the given data is not C-contiguous.
            If a sparse matrix is passed, a copy will be made if it's not in
            CSR format.


        Returns
        -------
        self : object
            Fitted estimator.
        """

        ####VALDIATING #####
        verbose = self.verbose
        X = super()._validate_data(X)

        super()._check_params_vs_input(X)

        random_state = get_random_state(self.random_state)

        # Validate init array
        init = self.init
        init_is_array_like = _is_arraylike(init)
        if init_is_array_like:
            init = check_array(init, dtype=X.dtype, copy=True, order="C")
            super()._validate_center_shape(X, init)

        # Todo add lsh Algorithm Here
        # if self._algorithm == "lsh++":
        #     pass
        # else:
        #     pass

        ###### KMEANS ALGORITHM #################################

        n_points, n_attrs = X.shape

        results = []
        seeds = random_state.randint(np.iinfo(np.int32).max, size=self.n_init)
        for init_no in range(self.n_init):
            start_initialization_centroid = time.time()
            print("Init: initializing centers")
            centroids = super()._init_centroids(
                X,
                init=init,
                random_state=random_state,
            )
            if self.save == 'timing' or self.save == 'data':
                rtime = {'step': 'initalization', 'init_no': init_no + 1,
                         'step_time': time.time() - start_initialization_centroid}
                with open(f'{self.save_dir}/{init_no}_centers_inital.json', 'w') as f:
                    f.write(json.dumps(rtime))

            results.append(_mhkmeans(
                X, lsh, self.n_clusters, n_points, self.max_iter, self.dissimilarity, centroids, init_no,
                verbose, seeds[init_no], self.save, self.save_dir
            ))

        all_centroids, all_labels, all_costs, all_n_iters, all_epoch_costs = zip(*results)

        best = np.argmin(all_costs)
        if self.n_init > 1 and verbose:
            print(f"Best run was number {best + 1}")
        self.cluster_centers_ = all_centroids[best]
        self.labels_ = all_labels[best]
        self.inertia_ = all_costs[best]
        self.cost_ = all_costs[best]
        self.n_iter_ = all_n_iters[best]
        self.epoch_costs_ = all_epoch_costs[best]

        return self
