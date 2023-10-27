import random


def simple_callback_random_permutation(prime):
    a = random.randint(1,prime - 1)
    b = random.randint(1,prime - 1)
    return a, b


class RandomPermutation:
    def __init__(self, type: str = 'type1', prime: int = 4294967311):
        if type == 'type1':  # TODO Mkae
            self.generate = self._generator1
        elif type == 'type2':
            self.generate = self._generator2
        elif type == 'callback':
            self.generate = self.call_back_generator
        self.prime = prime

    def call_back_generator(self, N: int):
        a = random.randint(1, self.prime - 1)
        b = random.randint(1, self.prime - 1)
        p = self.prime

        def call_back_generate_p(vector_index: int):
            return (a * vector_index + b) % p % N

        return call_back_generate_p

    def _generator1(self, N: int):
        '''
        function code to generate of 1 to N using big prime and two random numbers
        computing formula `f(i)=(a*i+b) mod prime mode N`
        for each element `i` in range `0` to `N`
        where the time complexity is O(N)
        :param N: number of elements in permutation
        :return: a list describing a permutation 0 to N-1
        '''

        a = random.randint(1, self.prime - 1)
        b = random.randint(1, self.prime - 1)
        permutation = [(a * i + b) % self.prime % N for i in range(N)]
        return permutation

    def _generator2(self, N: int):
        '''
                function to generate N numbers with  using big prime and two random numbers
                computing formula `f(i)=(a*i+b) mod prime`
                for each element `i` in range `0` to `N`
                :param N: number of elements in permutation
                :return: a list contains N numbers with random order
                '''
        a = random.randint(1, self.prime - 1)
        b = random.randint(1, self.prime - 1)
        permutation = [(a * i + b) % self.prime for i in range(N)]
        return permutation
