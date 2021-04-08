import numpy as np
np.set_printoptions(precision=3,suppress=True)
import simil

source_points = [[0, 0, 0],
                 [0, 2, 2],
                 [2, 3, 1]]

target_points = [[3, 7, 5],
                 [6, 7, 2],
                 [4.5, 4, 0.5]]

source_points_array = np.array(source_points)
target_points_array = np.array(target_points)
source_coords = source_points_array.T
target_coords = target_points_array.T
n = source_coords.shape[1]
alpha_0 = np.ones(n)
lambda_0=1.0
scale = True

# Eq 29
source_q_coords = np.concatenate((source_coords,np.zeros((1,n))))
print('\nsource_q_coords :\n', source_q_coords)
target_q_coords = np.concatenate((target_coords,np.zeros((1,n))))
print('\ntarget_q_coords :\n', target_q_coords)

# Eq 40
b_scalar = simil._get_scalar(alpha_0, source_q_coords)
print('\nb_scalar : ', b_scalar)

# Eq 41
c_scalar = simil._get_scalar(alpha_0)
print('\nc_scalar : ', c_scalar)

# Eq 17
q0_w_matrix = simil._get_w_matrix(source_q_coords.T)
print('\nq0_w_matrix :\n', q0_w_matrix)

# Eq 16
qt_q_matrix = simil._get_q_matrix(target_q_coords.T)
print('\nqt_q_matrix :\n', qt_q_matrix)

# Eq 42
a_matrix = simil._get_abc_matrices(alpha_0, q0_w_matrix, qt_q_matrix)
print('\na_matrix :\n', a_matrix)

# Eq 43
b_matrix = simil._get_abc_matrices(alpha_0, qt_q_matrix)
print('\nb_matrix :\n', b_matrix)

# Eq 44
c_matrix = simil._get_abc_matrices(alpha_0, q0_w_matrix)
print('\nc_matrix :\n', c_matrix)

# Initiation
lambda_i, i = lambda_0 , 1

# Iteration
blc_matrix, d_matrix, beta_1, r_quat, lambda_i, i = simil._get_solution(a_matrix,
                                                                   b_scalar, 
                                                                   b_matrix, 
                                                                   c_scalar, 
                                                                   c_matrix, 
                                                                   scale, 
                                                                   lambda_i, 
                                                                   i)
print('\nblc_matrix :\n', blc_matrix)
print('\nd_matrix :\n', d_matrix)
print('\nbeta_1 : ', beta_1)
print('\nr_quat :\n', r_quat)
print('\nlambda_i : ', lambda_i)
print('\ni : ', i)

# Eq 25
## Inermediate results
r_w_matrix = simil._get_w_matrix([r_quat])[0]
print('\nr_w_matrix :\n', r_w_matrix)
r_q_matrix = simil._get_q_matrix([r_quat])[0]
print('\nr_q_matrix :\n', r_q_matrix)
r_matrix = (r_w_matrix.T @ r_q_matrix)[:3,:3]
print('\nr_matrix :\n', r_matrix)
## From private function
#r_matrix = simil._get_r_matrix(r_quat)
#print('\nr_matrix :\n', r_matrix)

# Eq 58
s_quat = simil._get_s_quat(c_scalar, blc_matrix, r_quat)
print('\ns_quat :\n', s_quat)

# Eq 28, 26
t_vector = np.array(simil._get_t_vector(r_quat, s_quat)).reshape(3,1)
print('\nt_vector :\n', t_vector)

# Outputs
m = lambda_i
r = r_matrix
t = t_vector

# Verification
target_computed_points = (m * r @ source_coords + t).T
print('\ntarget_computed_points :\n', target_computed_points)
