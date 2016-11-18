import tarfile
import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize

import time
import random


def calculate_consine_diffs(feature_matrix):
    return 1 - (np.cosine(np.dot(feature_matrix, 
                                feature_matrix.transpose())/(np.sqrt(np.sum(np.power(feature_matrix,2)))*np.sqrt(np.power(feature_matrix, 2))))/np.pi)


#
#   Returns a matrix which columns corresponds to a specific feature:
#   Each row corresponds to a song
#   Each field a for the moment floats
#
#   features:A list containing features
#   dataframe: frame containing all feature data
#   n: number of songs
#
#   returns: Numpy.Matrix(col=feature,row=songs)
#
def extract_fields(features, dataframe, n, current_index, bands):
    number_of_features = len(features)
    
    feature_data_matrix = np.empty(())
    '''feature_data_matrix = np.empty((n, number_of_features))
    for i in range(n):
        col_index = 0
        for feature in features:
            feature_data_matrix[i][col_index] = dataframe.iloc[i][feature]
            col_index += 1'''

    i = current_index
    while i < current_index+bands:
        j = 0
        col_index = 0
        for feature in features:
            feature_data_matrix[j][col_index] = dataframe.iloc[i][feature]
            col_index += 1
        i += 1
        j += 1

    print(feature_data_matrix)
    
    # Is this correct?
    feature_data_matrix = normalize(feature_data_matrix, norm='l2', axis=0)

    return feature_data_matrix

def extract_fields_full(features, dataframe, n):
    number_of_features = len(features)
    feature_data_matrix = np.empty((n, number_of_features))
    for i in range(n):
        col_index = 0
        for feature in features:
            feature_data_matrix[i][col_index] = dataframe.iloc[i][feature]
            col_index += 1

    #print(feature_data_matrix)
    
    time1 = time.time()
    feature_data_matrix = normalize(feature_data_matrix, norm='l2', axis=0)
    time2 = time.time()

    print("L2 norm method time: ", time2 - time1)
    
    return feature_data_matrix

#
#   Generates a "random" matrix
#
#   Not sure if this is what we looking for.
#
def generate_random_v(rows, cols):
    seed = time.time()

    v = np.empty((rows, cols))

    random.seed(seed) 
    for i in range(rows):
        for j in range(cols):
            v[i][j] = random.choice([-1, 1])

    #print(v)
    return v

def cosine_distance(song1, song2):
    return 1 - np.cos(np.dot(song1, song2) * (1 / (np.sqrt(np.sum(song1**2))*np.sqrt(np.sum(song2**2)))))/np.pi


def find_duplicates(feature_data_matrix, r, b, sigma):
    dimensions = feature_data_matrix.shape

    time1 = time.time()
    v = generate_random_v(256, dimensions[1])
    time2 = time.time()
    print("Time taken to generate random V: ", time2-time1)
 
    print("Rank of matrix: ", np.linalg.matrix_rank(v))
    #   Uncomment to show v
    #print("Showing the v matrix: ")
    #print(v)
    
    #   Uncomment to show relations
    #   print("Showing relation_matrix: ")
    signature_matrix = np.dot(feature_data_matrix, v.transpose())
    print(signature_matrix)
    print("signature_matrix:s shape: ", signature_matrix.shape)

    signatures = np.sign(signature_matrix).transpose()
    print("Signatures output : ", signatures)

    start_band_index = 0
   
    band_gap = int(round(signatures.shape[0]/b))
    
    print("Gap: ", band_gap)

    end_band_index = start_band_index + band_gap 

    buckets = dict()
    
    while end_band_index < signatures.shape[0]:
        print("Start index : ", start_band_index, " and end index: ", end_band_index)

        band = signatures[start_band_index:end_band_index]
        orig_number_of_rows = band.shape[0]

        print("Band:")
        number_of_rows = band.shape[0]
         
        l = []
        for j in range(band.shape[1]):
            t = [band[i][j] for i in range(number_of_rows)]
            hashed_column = hash(tuple(t))
            if hashed_column not in buckets:
                buckets[hashed_column] = [(start_band_index, end_band_index)]
            else:
                buckets[hashed_column].append((start_band_index, end_band_index))

        #print([tuple([band[i][j] for i in range(number_of_rows)]) for j in range(band.shape[1])])
        
        start_band_index = end_band_index+1
        end_band_index += band_gap

    n_buckets = dict()
    n_buckets = {k:v for k,v in buckets.items() if len(v) > 1}
   
    print("Orig buckets: ", len(buckets))
    print("Reduced buckets: ", len(n_buckets))
    #print("Buckets: ", buckets)

    '''feature_data_matrix_transpose = feature_data_matrix.transpose()

    for intervals in n_buckets.values():
        for i in intervals:
            start = i[0]
            end = i[1]
            
            while start < end:
                temps = start + 1
                print("temps ", temps, " start: ", start)
                while temps < end:
                    if cosine_distance(feature_data_matrix[start], feature_data_matrix[temps]) <= sigma:
                        print("Found potential neighbour")
                    temps +=1
                start += 1'''

    return 0


#find_duplicates(np.empty((1000000, 31)), 20, 3, 5)

t = tarfile.open('millionsongsubset_full.tar.gz', "r:gz")
members = t.getmembers()

#for m in members:
#    print(m.name)

'''t.extract(members[7])
df = pd.read_csv(members[7].name, sep='<SEP>', engine='python', header=None)
print(df)'''

#np.unique(df[0], return_counts=True)
t.extract(members[5].name)
summary = pd.HDFStore(members[5].name)
print("Extracting the features...")
time1 = time.time()
feature_data_matrix = extract_fields_full(['duration', 
                'end_of_fade_in', 
                'key', 
                'loudness', 
                'mode', 'start_of_fade_out',
                'tempo',
                'time_signature'], summary['analysis/songs'], 1000)
time2 = time.time()
print("Real time elapsed for extract fields: ", time2-time1)
time1 = time.time()
find_duplicates(feature_data_matrix, 64, 3, 2)
time2 = time.time()

print("Time taken to find duplicates: ", time2-time1)

t.close()
#print("Type summary: ", type(summary['analysis/songs']))
#print("Some field :", summary['analysis/songs'].iloc[0])

#print("Type of summary field: ", type(summary['analysis/songs'].iloc[0]))

#print("Duration field value: ", summary['analysis/songs'].iloc[0]['duration'])

#print(summary['/analysis/songs'])
