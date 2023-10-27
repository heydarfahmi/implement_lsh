import numpy as np
from .util.dissim import matching_dissim, ng_dissim

class LshKmodes:

    def __init__(self,n_clusters=8,max_iteration=100,dissimilarity,
                 init='random',n_init=10,**kwargs):
        self.n_clusters = n_clusters
        self.max_iter = max_iteration
        self.dissimilarity = dissimilarity
        self.init = init
        self.random_state = random_state

    def fit(self,X,y=None,**kwargs):
        pass
    def fit_predict(self, X, y=None, **kwargs):
        """Compute cluster centroids and predict cluster index for each sample.

        Convenience method; equivalent to calling fit(X) followed by
        predict(X).
        """
        return self.fit(X, **kwargs).predict(X, **kwargs)

    def predict(self,X,**kwargs):
        pass
    @property
    def cluster_centroids_(self):
        pass



def _k_modes_single(X, n_clusters, n_points, n_attrs, max_iter, dissim, init, init_no,
                    verbose, random_state, sample_weight=None):
    random_state = check_random_state(random_state)
    # _____ INIT _____
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
        assert init.shape[1] == n_attrs, \
            f"Wrong number of attributes in init ({init.shape[1]}, " \
            f"should be {n_attrs})."
        centroids = np.asarray(init, dtype=np.uint16)
    else:
        raise NotImplementedError

    if verbose:
        print("Init: initializing clusters")
    membership_matrix = np.zeros((n_clusters, n_points), dtype=np.bool_)
    # cl_attr_freq is a list of lists with dictionaries that contain the
    # frequencies of values per cluster and attribute.
    cl_attr_freq = [[defaultdict(int) for _ in range(n_attrs)]
                    for _ in range(n_clusters)]
    for ipoint, curpoint in enumerate(X):
        weight = sample_weight[ipoint] if sample_weight is not None else 1
        # Initial assignment to clusters
        clust = np.argmin(dissim(centroids, curpoint, X=X, membship=membship))
        membship[clust, ipoint] = 1
        # Count attribute values per cluster.
        for iattr, curattr in enumerate(curpoint):
            cl_attr_freq[clust][iattr][curattr] += weight
    # Perform an initial centroid update.
    for ik in range(n_clusters):
        for iattr in range(n_attrs):
            if sum(membship[ik]) == 0:
                # Empty centroid, choose randomly
                centroids[ik, iattr] = random_state.choice(X[:, iattr])
            else:
                centroids[ik, iattr] = get_max_value_key(cl_attr_freq[ik][iattr])

    # _____ ITERATION _____
    if verbose:
        print("Starting iterations...")
    itr = 0
    labels = None
    converged = False

    _, cost = labels_cost(X, centroids, dissim, membship, sample_weight)

    epoch_costs = [cost]
    while itr < max_iter and not converged:
        itr += 1
        centroids, cl_attr_freq, membship, moves = _k_modes_iter(
            X,
            centroids,
            cl_attr_freq,
            membship,
            dissim,
            random_state,
            sample_weight
        )
        # All points seen in this iteration
        labels, ncost = labels_cost(X, centroids, dissim, membship, sample_weight)
        converged = (moves == 0) or (ncost >= cost)
        epoch_costs.append(ncost)
        cost = ncost
        if verbose:
            print(f"Run {init_no + 1}, iteration: {itr}/{max_iter}, "
                  f"moves: {moves}, cost: {cost}")

    return centroids, labels, cost, itr, epoch_costs
