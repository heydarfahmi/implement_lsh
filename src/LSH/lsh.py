MAX_NUMBER = 1000000000


class LSHDistance:
    def __init__(self, b, r):
        self.b = b
        self.r = r

    def split_vector(self, signature):
        b=self.b
        assert len(signature) % b == 0
        r = int(len(signature) / b)
        # code splitting signature in b parts
        subvecs = []
        for i in range(0, len(signature), r):
            subvecs.append(signature[i: i + r])
        return subvecs

    def sim(self, sig1: list, sig2: list) -> float:
        band1=self.split_vector(sig1)
        band2=self.split_vector(sig2)

        for rows1, rows2 in zip(band1, band2):
            if rows1 == rows2:
                pass