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

    print("Highest playcount: ", highest_playcount)

    return (users, songs, play_count)

if len(sys.argv) < 2:
    print("To few arguments...")
    sys.exit(1)

print("Parsing the triplets...")
users, songs, play_count = parse_triplets(sys.argv[1], 300000, False, 10, True, False)

# Easiest way to create a sparse matrix is done by using the vectors
resulting_sparse_matrix = create_sparse_matrix(users, songs, play_count, row=True)
print("Done parsing the triplets -> Sparse matrix")
sys.stdout.flush()
