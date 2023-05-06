from src.utils.permutation import RandomPermutation
generator = RandomPermutation()
for i in range(10):
    print(generator.generate(10))
