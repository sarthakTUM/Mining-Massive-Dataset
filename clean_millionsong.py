import tarfile
import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.preprocessing import scale

import time
import random

path_to_dataset = 'millionsongsubset_full.tar.gz'

#
#   Extract fields
#
def extract_fields_full(features, dataframe, n):
    number_of_features = len(features)
    feature_data_matrix = np.empty((n, number_of_features))
    for i in range(n):
        col_index = 0
        for feature in features:
            feature_data_matrix[i][col_index] = dataframe.iloc[i][feature]
            col_index += 1
    
    time1 = time.time()
    feature_data_matrix = scale(feature_data_matrix)
    #feature_data_matrix = normalize(feature_data_matrix, norm='l2', axis=0)
    time2 = time.time()

    print("L2 norm method time: ", time2 - time1)
    
    return feature_data_matrix

def cosine_distance(s1, s2):
    mag1 = np.linalg.norm(s1)
    mag2 = np.linalg.norm(s2)
    
    #print("Types: ", type(mag1), " and ", type(mag2))
    return 1 - (np.arccos(np.dot(s1, s2) / (mag1*mag2))/np.pi)

def generate_random_vectors(rows, cols):
    seed = time.time()

    v = np.empty((rows, cols))

    random.seed(seed)
    for i in range(rows):
        for j in range(cols):
            v[i][j] = random.choice([0,1])

    return v

def find_candidates(songs, r, b, epsilon):
    features = songs.shape[1]
    
    V = generate_random_vectors(r*b, features)

    hash_matrix = np.sign(np.dot(songs, V.transpose()))
    
    print("Hash matrix: ", hash_matrix)
    
    buckets = dict()
    random_hash = np.matrix([[2**(i+1)] for i in range(r)])
    hash_matrix_dimension = hash_matrix.shape

    print("Hash matrix dimensions: ", hash_matrix_dimension)

    band_buckets = [dict() for i in range(b)]

    for i in range(hash_matrix_dimension[0]):
        band_index = 0
        band = 0
        while band_index < hash_matrix_dimension[0]:
            print("Band index: ", band_index, " and ", band_index+r)
            hashed_value = np.dot([hash_matrix[i][band_index:band_index+r]], random_hash)[0][0].item()
            if hashed_value not in band_buckets[band]:
                band_buckets[band][hashed_value] = [i]
            else:
                band_buckets[band][hashed_value].append(i)
            band_index += r
            band += 1
    
    for b in band_buckets:
        print("Band buckets :", len(b))

    


t = tarfile.open(path_to_dataset, 'r:gz')
members = t.getmembers()

t.extract(members[5].name)
summary = pd.HDFStore(members[5].name)
print('Extracting the features...')

songs = extract_fields_full(['duration',
                            'end_of_fade_in',
                            'key',
                            'loudness',
                            'mode',
                            'start_of_fade_out',
                            'tempo',
                            'time_signature'], summary['analysis/songs'], 9999)
time1 = time.time()
buckets = find_candidates(songs, 64, 3, 2)
time2 = time.time()
print("Time taken to find candidates: ", time2 - time1)

