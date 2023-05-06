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


class BooleanShingler:

    def __init__(self, k, path=None):
        self.k = k
        self.shingle_set = ShingleSet()
        self.shingle_index = set()
        self.document_shingles = []
        if path:
            self.load_shingle()

    def boolean_vector_shingling(self, documents):
        """
            Implements k-gram word-based shingling for a list of texts and k value.
            Returns a list of text set ,boolean vectors and shingles and all words set.
            """
        k = self.k
        shingle_set = ShingleSet()  # set to store all shingles
        text_sets = []
        vectors = []
        for text in documents:
            words = text.split()
            text_set = set()
            boolean_vector = set()
            for i in range(len(words) - k + 1):
                shingle = ' '.join(words[i:i + k])
                text_set.add(shingle)
                boolean_vector.add(shingle_set.add(shingle))
            text_sets.append(text_set)
            vectors.append(list(boolean_vector))

        return text_sets, vectors, shingle_set

    def kgram_shingling_single_text(self, text):
        """
        Implements k-gram word-based shingling for a given text and k value.
        Returns a set of shingles.
        """
        k = self.k
        words = text.split()  # split text into words
        shingles = set()  # set to store shingles

        # generate k-grams and add them to the set
        for i in range(len(words) - k + 1):
            shingle = ' '.join(words[i:i + k])
            shingles.add(shingle)

        return shingles

    def save_shingle(self):
        pass

    def load_shingle(self):
        pass
