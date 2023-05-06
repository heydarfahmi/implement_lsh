MIN_NUMBER = -10  # TODO


class Kmeans:

    def __init__(self, vectors, vectors_num):
        self.vectors = vectors
        self.vectors_len = vectors_num

    def set_similarity(self, callback_similarity):
        self.cal_sim = callback_similarity

    def set_initial_state(self, callback_initial_state):
        self.initial_state = callback_initial_state

    def run_kmean(self, run_time,**kwargs):
        for i in range(run_time):
            self.update_group(**kwargs)

    def update_groups(self, k, centers, groups, max_sim):
        centers = self.initial_state(k, self.vectors)
        groups = [0 for i in range(self.vectors_len)]
        max_sim = [MIN_NUMBER for i in range(self.vectors_len)]
        for c_index, center in enumerate(centers):
            for v_index, vector in enumerate(self.vectors):
                cv_distance = self.cal_sim(center, vector)

                if cv_distance > max_sim[v_index]:
                    max_sim[v_index] = cv_distance
                    groups[v_index] = c_index
