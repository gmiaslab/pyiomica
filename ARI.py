import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import adjusted_rand_score

def runForClusterNum(arguments):

    np.random.seed()

    cluster_num, data_array, trials_to_do = arguments

    print(cluster_num, end=', ', flush=True)

    labels = [KMeans(n_clusters=cluster_num).fit(data_array).labels_ for i in range(trials_to_do)]

    agreement_matrix = np.zeros((trials_to_do,trials_to_do))

    for i in range(trials_to_do):
        for j in range(trials_to_do):
            agreement_matrix[i, j] = adjusted_rand_score(labels[i], labels[j]) if agreement_matrix[j, i] == 0 else agreement_matrix[j, i]

    selected_data = agreement_matrix[np.triu_indices(agreement_matrix.shape[0],1)]

    return np.array((cluster_num, np.mean(selected_data), np.std(selected_data)))