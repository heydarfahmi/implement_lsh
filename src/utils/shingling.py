from src.utils.storage import ShingleSet


class BooleanShingler:

    def __init__(self, k, path=None):
        self.k = k
        self.shingle_set = ShingleSet()
        self.shingle_index = set()
        self.document_shingles = []

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
        self.shingle_set = shingle_set
        self.document_shingles = vectors
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
