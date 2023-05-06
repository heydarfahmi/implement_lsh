from src.utils.shingling import BooleanShingler
shingler=BooleanShingler(2,None)
texts = ["the quick brown fox jumps over the lazy dog",
         "the quick brown fox jumps over the lazy cat",
         "the quick brown dog jumps over the lazy fox",
         "the quick brown cat jumps over the lazy fox"]

t,b,s=shingler.boolean_vector_shingling(texts)
for i,tt in enumerate(t) :
    print(tt)
    print(b[i])
print(s.shingle_list)


