MAX_NUMBER = 1000000000


class MinHash:
    def __init__(self, Documents=None):
        self.dynamic_matrix = Documents
        self.matrix_signature = None

    def sign_dynamic_matrix(self, permutations):
        '''
        sign documents using minhash algorithm base on
        permutations given as input.
        :param permutations: a list permutations or hashes of words' orders
        :return: 2D array containing the signature of the class matrix
        '''
        matrix_signature = []
        for column in self.dynamic_matrix:
            column_signature = []
            for p in permutations:
                column_signature.append(
                    self._advance_sign_column(column, p)
                )
            matrix_signature.append(column_signature)
        self.matrix_signature = matrix_signature
        return matrix_signature

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
