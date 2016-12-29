import re
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl
import sys
import matplotlib.pyplot as plt
import random
import time
import math
from scipy import sparse
from sklearn.linear_model import Ridge

# The sparse matrix function
def create_sparse_matrix(users, songs, play_count, row=True):
    if row:
        return sp.csr_matrix((play_count, (users, songs)), shape=(len(users), len(songs)))
    else:
        return sp.csc_matrix((play_count, (users, songs)), shape=(len(users), len(songs)))

# The parse triplets function
def parse_triplets(file_path, max_rows, whole_dataset, b, use_vectors, use_dikt):
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

        if int(triplet_match.group(3)) > highest_playcount:
            highest_playcount = int(triplet_match.group(3))

        current_row += 1
        if current_row >= max_rows and not whole_dataset:
            break

    triplets.close()

    print("Highest playcount: ", highest_playcount)

    return (users, songs, play_count)

    
def binning(play_count, bin_size):
    bin_array = [2**i for i in range(bin_size)]
    print("Bin Array : ", bin_array)
    binned_counts = np.digitize(play_count, bin_array, right = False)
    print("Binned Counts : ", binned_counts)
    return binned_counts
        
    
def cold_start(user_song_matrix, min_threshold):
    #We need to remove songs & users with less or equal to min_threshold 
    #print(user_song_matrix)
    user_song_matrix = user_song_matrix > 0
    user_song_matrix = user_song_matrix.astype(np.int)
    #print(user_song_matrix.toarray())
        
    songs_per_user = np.ravel(np.sum(user_song_matrix, axis=1)) #sum of row
    users_to_delete = np.where(songs_per_user <= min_threshold)[0]
    
    ##set the users i.e rows to 0
    for i in users_to_delete:
            user_song_matrix.data[user_song_matrix.indptr[i]:user_song_matrix.indptr[i+1]]=0

    user_song_matrix.eliminate_zeros()
    mask = np.concatenate(([True], user_song_matrix.indptr[1:] != user_song_matrix.indptr[:-1]))
    user_song_matrix = sp.csr_matrix((user_song_matrix.data, user_song_matrix.indices, user_song_matrix.indptr[mask]))

    ##set the songs i.e columns to 0
    user_song_matrix = sp.csc_matrix(user_song_matrix)
    
    users_per_song = np.ravel(np.sum(user_song_matrix, axis=0)) #sum of column
    songs_to_delete = np.where(users_per_song <= min_threshold)[0]

    for i in songs_to_delete:
            user_song_matrix.data[user_song_matrix.indptr[i]:user_song_matrix.indptr[i+1]]=0

    user_song_matrix.eliminate_zeros()
    mask = np.concatenate(([True], user_song_matrix.indptr[1:] != user_song_matrix.indptr[:-1]))
    user_song_matrix = sp.csc_matrix((user_song_matrix.data, user_song_matrix.indices, user_song_matrix.indptr[mask]))	
    
    return user_song_matrix.tocsr()

def singular_value_decomp(sparse_matrix):
    sparse_matrix = sparse_matrix.asfptype()
    print("Matrix shape : ", sparse_matrix.shape)
    # print("Sparse Matrix : ", sparse_matrix)
    
    U, sigma, Vt = spl.svds(sparse_matrix, k=30)
    print("U shape : ", U.shape, " sigma shape : ", sigma.shape, " V shape : ", Vt.shape)
    
    sigma = np.diag(sigma)
    print("Sigma shape after converting to diagonal matrix : ", sigma.shape)
    Q = U.dot(sigma)
    print("Q shape : ", Q.shape)
    print("Pt shape : ", Vt.shape)
    return Q, Vt

#
#   This functions picks a random set of size: number_of_random_elements
#   
#   Returns: M_s the modified M
#            random_set containing all random pciked elements on the form [(row,col,data)]
#
def pick_random_test_set(M, number_of_random_elements):
    random.seed(time.time())
    
    M_s = sp.coo_matrix(M)
    interval = len(M_s.data)

    random_picked_values = []
    already_picked_values = []
    
    for i in range(number_of_random_elements):
        random_index = random.randint(0, interval-1)

        while random_index in already_picked_values:
            random_index = random.randint(0, interval-1)
        already_picked_values.append(random_index)
        
        random_picked_values.append((M_s.row[random_index], 
                                    M_s.col[random_index], 
                                    M_s.data[random_index]))
        M_s.data[random_index] = 0

    M_s.eliminate_zeros()
    M_s = sp.csr_matrix(M_s)
    return M_s, random_picked_values

def calc_loss(M, Q, Pt):
    loss = 0
	#TODO we have to optimize it
    lamda1 = 0.5;lamda2 = 0.5
    M = M.tocoo()
    print("size of data", len(M.data)); print("size of row", len(M.row)) 
    print("size of col", len(M.col))
	
    for d in range (len(M.data)):
        r = M.data[d]; x = M.row[d]; i = M.col[d];
        #print("r x i d", r,x,i,d)
		#TODO FIXME which one is correct ?
        #loss += math.pow((r - (np.dot(Q[i,:],Pt[:,x]))),2)
        loss += math.pow((r - (np.dot(Q[x,:],Pt[:,i]))),2)
    loss = loss + np.sum(np.square(np.linalg.norm(Pt, axis=0))) + np.sum(np.square(np.linalg.norm(Q, axis=1)));
    print("loss", loss)
    M = M.tocsr()
    return loss

def gradient_loss_px(M, Q, Pt, i):
	gradient = 0; lamda1 = 0.5;
	i_indices = np.where(M.col == i)[0]
	#print("i_indices length", len(i_indices), "value", M.col[i_indices[0]], M.col[i_indices[1]], M.col[i_indices[2]], i)
	for i_1 in i_indices:
	    r = M.data[i_1]; x = M.row[i_1]; i1 = M.col[i_1];
	    gradient += (r - (np.dot(Q[x,:], Pt[:,i1]))) * np.array(Q[x,:])
	gradient = -2 * gradient
	#print("gradientP shape", gradient.shape)
	gradient += 2*lamda1*(np.sum(Pt, axis=1))
	#print("gradientP shape", gradient.shape)
	return gradient

def gradient_loss_qi(M, Q, Pt, x):
	gradient = 0; lamda2 = 0.5;
	x_indices = np.where(M.row == x)[0]
	#print("x_indicies", x_indices)
	for x_1 in x_indices:
	    r = M.data[x_1]; x1 = M.row[x_1]; i = M.col[x_1];
	    gradient += (r - (np.dot(Q[x1,:], Pt[:,i]))) * np.array(Pt[:,i])
	gradient = -2 * gradient
	#print("gradientQ shape", gradient.shape)
	gradient += 2*lamda2*(np.sum(Q, axis=0))
	#print("gradientQ shape", gradient.shape)
	return gradient

def gradient_descent(M, Pt, Q):
    lr_rate = 0.3
    print("M size", M.shape)
    print("Q[0,:] size", Q[0,:].shape)
    print("Pt[:,0] size", Pt[:,0].shape)
    print("num of element in M", len(M.data))
    loss1 = calc_loss(M, Q, Pt)
    print("P shape index", Pt.shape[0], Pt.shape[1])
    M = M.tocoo()
    for i in range(Pt.shape[1]):
       Pt[:, i] = Pt[:, i] - lr_rate * gradient_loss_px(M, Q, Pt, i)
    for x in range(Q.shape[0]):
       Q[x,:] = Q[x,:] - lr_rate * gradient_loss_qi(M, Q, Pt, x)
    print("Q size", Q.shape)
    print("Pt size", Pt.shape)
    M = M.tocsr()
    loss2 = calc_loss(M, Q, Pt)
    print("loss1", loss1, "loss2", loss2)
    return None

def original_stochastic_gradient_descent(M, P, Q):
    return None

def original_stochastic_gradient_descent_larger_batches(M, P, Q):
    return None

def finding_method(M, Q, P):
    P = sp.csr_matrix(P)
    Q = sp.csr_matrix(Q)
    M = sp.csr_matrix(M)

    print("P sh:", P.shape)
    print("Q sh:", Q.shape)
        
    m_shape = M.shape
    
    print(type(M[0,:]))

    print("Q.T shape: ", Q.T.shape)
    print("Q shape: ", Q.shape)
    #print(np.einsum('ji,j->ji', Q, Q))

    qTq = np.sum(np.dot(Q.T,Q).diagonal())
    print("qTq: ", qTq)
    #print(qTq.shape)

    pTp = np.sum(np.dot(P.T, P).diagonal())
    print("pTp: ", pTp)
    
    for x in range(m_shape[1]):
        m = M[:,x]
        #print("Q.T shape: ", Q.T.shape)
        #print("m.T shape: ",  m.T.shape)

        #print("Q shape: ", Q.shape)
        #print("m shape: ",  m.shape)
        p_x = (1/pTp) * np.dot(Q.T, m)
        
        #print("P shape first : ", P.shape)
        P = sp.vstack((sp.vstack((P[:x], p_x.T)), P[x+1:]))
        #print("P shape after : ", P.shape)
        #print(P.data[P.indptr[x]:P.indptr[x+1]])

        #print("p_x: ", p_x.shape)

    print("Done with P")

    for i in range(m_shape[0]):
        m = M[i,:]
        q_i = (1/qTq) * np.dot(P.T, m.T)

        Q = sp.vstack((sp.vstack((Q[:i], q_i.T)), Q[i+1:]))

    print(np.dot(Q,P.T).shape)

    np.sum(np.norm('l2', M - np.dot(Q,P.T)), axis=1)
    #print(m.shape)
    #print(l.shape)
    print("Done")
    
def rmse(Mdiff, test_set):
    
    y_sum = 0
    for test in test_set:
        y_sum += (test[2] - Mdiff[test[0]][test[1]])**2

    rmse = np.sqrt(y_sum/len(test_set))

    print("RMSE: ", rmse)

if len(sys.argv) < 2:
    print("To few arguments...")
    sys.exit(1)

min_threshold = 5
bin_size = 10 ## This is configurable by user
size_of_test_set = 200

print("Parsing the triplets...")
users, songs, play_count = parse_triplets(sys.argv[1], 300000, False, 10, True, False)

# Here the logic for the binnig should be placed

binned_play_counts = binning(play_count, bin_size)

# Either pass this binned_play_counts to the following functions that require play_count
# as a parameter, or change the return variable "binned_play_counts" to "play_count"
# to override its value

# Easiest way to create a sparse matrix is done by using the vectors
resulting_sparse_matrix = create_sparse_matrix(users, songs, binned_play_counts, row=True)
print("Done parsing the triplets -> Sparse matrix")

shape_orig = resulting_sparse_matrix.shape
while True:
    shape_before = resulting_sparse_matrix.shape
    resulting_sparse_matrix = cold_start(resulting_sparse_matrix, min_threshold)
    shape_after = resulting_sparse_matrix.shape
    print("Intermediate shape: ", shape_after)
    if shape_before[0] == shape_after[0] and shape_before[1] == shape_after[1]:
        break
shape_final = resulting_sparse_matrix.shape

print("Picking random test set: ")
resulting_sparse_matrix, test_set = pick_random_test_set(resulting_sparse_matrix, size_of_test_set)

#print("Got random test set: ", test_set)

# Initial P & Q values obtained using SVD
Q, Pt = singular_value_decomp(resulting_sparse_matrix)

# Perform AO using P,Q
gradient_descent(resulting_sparse_matrix, Pt, Q)


#find_method(resulting_sparse_matrix, Q, Pt.T)
print("Orig shape: ", shape_orig, " Final shape: ", shape_after)
sys.stdout.flush()
