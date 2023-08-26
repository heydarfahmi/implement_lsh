MAX_NUMBER = 1000000000
from src.utils.permutation import RandomPermutation
import numpy as np


class MinHash:
    def __init__(self, n_permutations):
        self._matrix_signature = None
        self.n_permutations = n_permutations
        self._permutations = []

    def _advance_sign_column(self, column, p):
        '''
        sign a column base on min hash algorithm
        :param column:a list which contains the index of words in document
        :param p: a list defining universal hashes or permutation
        :return: signature of column
        '''
        # TODO get max NUMBER
        minhash_p = MAX_NUMBER
        for i in column:
            if p[i] < minhash_p:
                minhash_p = p[i]
        return minhash_p

    def _sign_coordinate_matrix(self, coordinate_matrix, permutations):
        '''
        sign documents using minhash algorithm base on
        permutations given as input.
        :param coordinate_matrix: a matrix containing  vectors  which each contains the nonzero index of
         binary  vector.
        :param permutations: a list permutations or hashes of bit orders
        :return: 2D array containing the signature of the class matrix
        '''
        matrix_signature = []
        for column in coordinate_matrix:
            column_signature = []
            for p in permutations:
                column_signature.append(
                    self._advance_sign_column(column, p)
                )
            matrix_signature.append(column_signature)
        return matrix_signature

    def sign_coordinate_matrix(self, coordinate_matrix, p_size):
        '''
       generate n permutations and
       sign documents using minhash algorithm base on them.
       :param coordinate_matrix: a matrix containing  vectors  which each contains the nonzero index of
         binary  vector.
        :return: 2D array containing the signature of the class matrix
       '''
        p_generator = RandomPermutation()
        self._permutations = [p_generator.generate(p_size) for _ in range(self.n_permutations)]
        self._matrix_signature = self._sign_coordinate_matrix(
            coordinate_matrix, self._permutations)

        return self._matrix_signature

    def sign_binary_matrix(self, binary_matrix):
        '''
       generate n permutations and
       sign documents using minhash algorithm base on them.
       :param binary_matrix: matrix containing vectors that each  emphasizes that the vector consists of
       only two distinct values, 0 and 1, which can represent binary choices or conditions.
       :return: 2D array containing the signature of the class matrix
       '''
        coordinate_matrix = [list(np.nonzero(document)) for document in documents]
        return self.sign_coordinate_matrix(
            coordinate_matrix,
            len(binary_matrix[0])
        )
