from collections import defaultdict
from abc import ABCMeta, abstractmethod

ABC = ABCMeta('ABC', (object,), {})


class Storage(ABC):
    '''Base class for key, value containers where the values are sequences.'''

    def __getitem__(self, key):
        return self.get(key)

    def __delitem__(self, key):
        return self.remove(key)

    def __len__(self):
        return self.size()

    def __iter__(self):
        for key in self.keys():
            yield key

    def __contains__(self, item):
        return self.has_key(item)

    @abstractmethod
    def keys(self):
        '''Return an iterator on keys in storage'''
        return []

    @abstractmethod
    def get(self, key):
        '''Get list of values associated with a key

        Returns empty list ([]) if `key` is not found
        '''
        pass

    def getmany(self, *keys):
        return [self.get(key) for key in keys]

    @abstractmethod
    def insert(self, key, *vals, **kwargs):
        '''Add `val` to storage against `key`'''
        pass

    @abstractmethod
    def remove(self, *keys):
        '''Remove `keys` from storage'''
        pass

    @abstractmethod
    def remove_val(self, key, val):
        '''Remove `val` from list of values under `key`'''
        pass

    @abstractmethod
    def size(self):
        '''Return size of storage with respect to number of keys'''
        pass

    @abstractmethod
    def itemcounts(self, **kwargs):
        '''Returns the number of items stored under each key'''
        pass

    @abstractmethod
    def has_key(self, key):
        '''Determines whether the key is in the storage or not'''
        pass

    def status(self):
        return {'keyspace_size': len(self)}


class DictListStorage(Storage):
    '''This is a wrapper class around ``defaultdict(list)`` enabling
    it to support an API consistent with `Storage`
    '''

    def __init__(self):
        self._dict = defaultdict(list)

    def keys(self):
        return self._dict.keys()

    def get(self, key):
        return self._dict.get(key, [])

    def remove(self, *keys):
        for key in keys:
            del self._dict[key]

    def remove_val(self, key, val):
        self._dict[key].remove(val)

    def insert(self, key, *vals, **kwargs):
        self._dict[key].extend(vals)

    def size(self):
        return len(self._dict)

    def itemcounts(self, **kwargs):
        '''Returns a dict where the keys are the keys of the container.
        The values are the *lengths* of the value sequences stored
        in this container.
        '''
        return {k: len(v) for k, v in self._dict.items()}

    def has_key(self, key):
        return key in self._dict


class DictSetStorage(DictListStorage):
    '''This is a wrapper class around ``defaultdict(set)`` enabling
    it to support an API consistent with `Storage`
    '''

    def __init__(self):
        self._dict = defaultdict(set)

    def get(self, key):
        return self._dict.get(key, set())

    def insert(self, key, *vals, **kwargs):
        self._dict[key].update(vals)


class ShingleSet:

    def __init__(self):
        self.shingle_set = {}
        self.shingle_list = []
        self.size = 0

    def add(self, object):
        result = self.shingle_set.setdefault(object, self.size)
        if result == self.size:
            self.size += 1
            self.shingle_list.append(object)
        return result

    def get_index(self, index):
        return self.shingle_list[index]

    def length(self):
        return self.size

    def get(self, object):
        return self.shingle_set.get(object)
