from src.utils.storage import DictSetStorage, DictListStorage


class LSH(object):
    '''
    The :ref:`minhash_lsh` index.
    Reference: `Chapter 3, Mining of Massive Datasets
    <http://www.mmds.org/>`_.

    Args:
        b:Lsh Parameter,number of bands
        r:Lsh Parameter, size of each band.
    '''

    def __init__(self, b, r):
        self.b = b
        self.r = r
        self._num_perm = r * b
        self._hash_func = self._byteswap
        self._hashranges = [(i * r, (i + 1) * r) for i in range(b)]
        self.hashtables = [
            DictSetStorage()
            for _ in range(self.b)]
        self.keys = DictListStorage()
        self._point_number = None

    def _insert(self, vector, key):
        Hs = [self._hash_func(vector[start:end])
              for start, end in self.hashranges]
        self.keys.insert(key, *Hs)
        for H, hashtable in zip(Hs, self.hashtables):
            hashtable.insert(H, key)

    def fit(self, minhash_matrix):
        '''
         hash  hash each vector(point) signature of Minhash matrix into buckets.
        the row number of each point its used as point identifier.

        :param minhash_matrix: its a min hash matrix which the columns belong to each point or document vecotr.
        '''
        self.keys = DictListStorage()
        for pid, pvector in enumerate(minhash_matrix):
            self._insert(pvector, pid)
        self._point_number = self.keys.size()

    def extend(self, minhash_matrix):
        '''
        update hash tables and buckets with new points.
        :param minhash_matrix:min hash matrix which the columns belong to each point or document vecotr.
        '''
        seek = self._point_numbers
        for pid, pvector in enumerate(minhash_matrix):
            self._insert(pvector, pid + seek)
        self._point_number = self.keys.size()

    def split_vector(self, signature):
        b = self.b
        assert len(signature) % b == 0
        r = int(len(signature) / b)
        # code splitting signature in b parts
        subvecs = []
        for i in range(0, len(signature), r):
            subvecs.append(signature[i: i + r])
        return subvecs

    def sim(self, sig1: list, sig2: list) -> float:
        band1 = self.split_vector(sig1)
        band2 = self.split_vector(sig2)

        for rows1, rows2 in zip(band1, band2):
            if rows1 == rows2:
                pass

    def _hash_func(self, hs):
        return bytes(hs.byteswap().data)

    def get_hash_counts(self):
        '''
        get number of buckets for each b
        '''
        counts = [
            hashtable.size() for hashtable in self.hashtables]
        return counts

    def get_buckets_size(self):
        '''
        Returns a list of length ``self.b`` with elements representing the
        number of keys stored under each bucket for the given permutation.
        '''
        counts = [
            hashtable.itemcounts() for hashtable in self.hashtables]
        return counts

    def is_empty(self):
        '''
        Returns:
            bool: Check if the index is empty.
        '''
        return any(t.size() == 0 for t in self.hashtables)

    def remove(self, pid):
        '''
        Remove the key from the index.

        Args:
            key (hashable): The unique identifier of a set.

        '''
        if pid not in self.keys:
            raise ValueError("The Point Id does not exist")
        for H, hashtable in zip(self.keys[pid], self.hashtables):
            hashtable.remove_val(H, pid)
            if not hashtable.get(H):
                hashtable.remove(H)
        self.keys.remove(pid)

    def query(self, vector_signature):
        '''
        Giving the minhash signature of the query set, retrieve
        the keys that reference sets with Jaccard
        similarities likely greater than the threshold.

        Results are based on minhash segment collision
        and are thus approximate. For more accurate results,
        filter again with `minhash.jaccard`. For exact results,
        filter by computing Jaccard similarity using original sets.

        :param vector_signature: The MinHash signature vector of the query set.

        Returns:
            `list` of unique keys.

        '''
        if len(vector_signature) != self._num_perm:
            raise ValueError("Expecting a vector with length %d, got %d"
                             % (self._num_perm, len(vector_signature)))
        candidates = set()
        for (start, end), hashtable in zip(self.hashranges, self.hashtables):
            _H = self._hash_func(vector_signature[start:end])
            for pid in hashtable.get(_H):
                candidates.add(pid)
        return list(candidates)

    def __contains__(self, key):
        '''
        Args:
            key :point identifier
        Returns:
            bool: True only if the key exists in the index.
        '''

        return key in self.keys
