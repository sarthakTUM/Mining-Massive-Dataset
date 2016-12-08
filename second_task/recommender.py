import re
import numpy as np
import scipy.sparse as sp
import sys
import matplotlib.pyplot as plt


# The sparse matrix function
def create_sparse_matrix(users, songs, play_count, row=True):
    if row:
        print("Create row")
        return sp.csr_matrix((play_count, (users, songs)), shape=(len(users), len(songs)))
    else:
        print("Create col")
        return sp.csc_matrix((play_count, (users, songs)), shape=(len(users), len(songs)))

# The parse triplets function
def parse_triplets(file_path, max_rows, whole_dataset, b, use_vectors, use_dikt):
    # DICTK

    some_dict = dict()

    # Vectors

    users = []
    songs = []
    play_count = []

    mapping_users = {}
    mapping_songs = {}
    
    # Code to extract users, songs and playcount from the train_triplets.txt 

    triplets = open(file_path, 'r')
    
    line_regex = re.compile("^([\w\d]+)\t([\w\d]+)\t([\w]+)$")

    current_row = 0

    highest_playcount = 0

    for triplet in triplets:
        triplet_match = line_regex.match(triplet)

        if (use_vectors):
            if triplet_match.group(1) not in mapping_users:
                mapping_users[triplet_match.group(1)] = len(mapping_users)
    
            if triplet_match.group(2) not in mapping_songs:
                mapping_songs[triplet_match.group(2)] = len(mapping_songs)

            users.append(int(mapping_users[triplet_match.group(1)]))
            songs.append(int(mapping_songs[triplet_match.group(2)]))
            play_count.append(int(triplet_match.group(3)))

        if (use_dikt):
            if triplet_match.group(3) not in some_dict:
                some_dict[triplet_match.group(3)] = [(triplet_match.group(1), triplet_match.group(2))]
        
        if int(triplet_match.group(3)) > highest_playcount:
            highest_playcount = int(triplet_match.group(3))

        current_row += 1
        if current_row >= max_rows and not whole_dataset:
            break

    triplets.close()

    # Binning the data:
    # Too slow method maybe?

    print("Hihgest playcount: ", highest_playcount)
   
    for i in range(len(play_count)):
        bin_value = 0
        while bin_value < b:
            if 2**(bin_value) <= play_count[i] < (2**(bin_value+1)-1):
                play_count[i] = bin_value
                break
            bin_value += 1


    return (users, songs, play_count)

def test_sum_rows(matrix):
    vector = matrix.sum(axis=1)
    print(vector)


def remove_lesser_occurance_of_songs_users(csr):
    #sum_over_columns = crc.sum(axis=0)

    shape_of_original = csr.shape
    ccm = csr
    while shape_of_original[0] != 0 and shape_of_original[1] != 0:
        sum_over_rows = ccm.sum(axis=1)
        element_list = list(ccm.toarray())
        
        

        for i in range(len(sum_over_rows)):
            if sum_over_rows[i] > 5:
                del element_list[i]
        
        ccm = sp.csc_matrix(np.array(element_list))
        sum_over_columns = ccm.sum(axis=0)

        element_list = list(ccm.toarray())

        for i in range(len(sum_over_columns)):
            if sum_over_columns[i] > 5:
                del element_list[i]
        
        ccm = sp.ccm_matrix(np.array(element_list))
        if shape_of_original[0] == ccm.shape[0] and shape_of_original[1] == ccm.shape[1]:
            break
    return ccm
    #print(sum_over_rows)
    #print(sum_over_columns)

def remove_row_in_LIL(matrix, i):
    matrix.rows = np.delete(matrix.rows, i)
    matrix.data = np.delete(matrix.data, i)
    matrix._shape = (matrix._shape[0]-1, matrix._shape[1])

def remove_lesser_occurance_of_songs_users_LIL(matrix):
    # Convert to LIL format
    lil_mat = sp.lil_matrix(matrix)
    orig_shape = lil_mat.shape

    while True:
        row_sums = lil_mat.sum(axis=1)
        for i in range(len(row_sums)):
            if row_sums[i] < 5:
                remove_row_in_LIL()


    return lil_mat

    print(rows)

if len(sys.argv) < 2:
    print("To few arguments...")
    sys.exit(1)

print("Parsing the triplets...")
users, songs, play_count = parse_triplets(sys.argv[1], 300000, False, 10, True, False)

# Easiest way to create a sparse matrix is done by using the vectors
resulting_sparse_row_matrix = create_sparse_matrix(users, songs, play_count, row=True)
resulting_sparse__column_matrix = create_sparse_matrix(users, songs, play_count, row=False)
print("Done parsing the triplets -> Sparse matrix")
#test_sum_rows(resulting_sparse_matrix)
#n_matrix = remove_lesser_occurance_of_songs_users(resulting_sparse_row_matrix)
n_matrix = remove_lesser_occurance_of_songs_users_LIL(resulting_sparse_row_matrix)
print(n_matrix.shape)
sys.stdout.flush()
