from  src.LSH.MinHash import MinHash

P = [[1, 2, 3, 4, 0], [3, 0, 2, 4, 1]]
basicC = [[1, 0, 1, 1, 0], [0, 1, 1, 0, 1]]
advanceC = [[0, 2, 3], [1, 2, 4]]
signature = MinHash(advanceC)

print(signature._advance_sign_column(advanceC[1], P[0]))


basicC=[[0,1,1,0],[1,0,1,1],[0,1,0,0]]
advanceC=[[1,2],[0,2,3],[1]]
signature=MinHash(advanceC)
p=[[0,1,2,3],[3,2,0,1]]
print(signature.sign_dynamic_matrix(p))
#the result should be [[1,1],[0,0],[1,3]]
#TODO write unit tests




basicC=[[1,1,0,0,0,1,1],[0,0,1,1,1,0,0],[0,1,1,1,1,0,0],[0,1,1,1,1,0,0]]
advanceC=[[0,1,5,6],[2,3,4],[0,5,6],[1,2,3,4]]
p=[[1,2,6,5,0,4,3],[3,1,0,2,5,6,4],[2,3,6,1,5,0,4]]
signature=MinHash(advanceC)
print(signature.sign_dynamic_matrix(p))
#result should be [[1, 1, 0], [0, 0, 1], [1, 3, 0], [0, 0, 1]]

#TODO write unit tests
