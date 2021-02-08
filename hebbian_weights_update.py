import numpy as np
from numba import njit

    

@njit
def forward_MLP2(X, W1, W2):
    A1 = np.tanh(W1 @ X)
    A2 = np.tanh(W2 @ A1)
    return A2.item()

@njit
def forward_MLP3(X, W1, W2, W3):
    A1 = np.tanh(W1 @ X)
    A2 = np.tanh(W2 @ A1)
    A3 = np.tanh(W3 @ A2 )
    return A3.item()

@njit
def forward_MLP3_withBias(X, W1, W2, W3, b1, b2, b3):
    A1 = np.tanh(W1 @ X  + b1.T)
    A2 = np.tanh(A1 @ W2 + b2.T)
    A3 = np.tanh(W3 @ A2.T  + b3)[0]
    return A3.item()

    
@njit
def neural_hebbian_update_ML3withBias(heb_coeffs, weights1_2, weights2_3, weights3_4, o0, o1, o2, o3):
    
        heb_offset = 0
        # Layer 1         
        for i in range(weights1_2.shape[1]): 
            for j in range(weights1_2.shape[0]):  
                idx = (weights1_2.shape[0]-1)*i + i + j
                
                W1 = heb_coeffs[idx][:6].reshape((3,2))
                W2 = heb_coeffs[idx][6:15].reshape((3,3))
                W3 = heb_coeffs[idx][15:18].reshape((1,3))
                b1 = heb_coeffs[idx][18:21].reshape((3,1))
                b2 = heb_coeffs[idx][21:24].reshape((3,1))
                b3 = heb_coeffs[idx][24]

                X = np.array([o0[i], o1[j]])
                
                weights1_2[:,i][j] += forward_MLP3_withBias(X, W1, W2, W3, b1, b2, b3)

                
        heb_offset = weights1_2.shape[1] * weights1_2.shape[0]
        # Layer 2
        for i in range(weights2_3.shape[1]): 
            for j in range(weights2_3.shape[0]):  
                idx = heb_offset + (weights2_3.shape[0]-1)*i + i+j
                
                W1 = heb_coeffs[idx][:6].reshape((3,2))
                W2 = heb_coeffs[idx][6:15].reshape((3,3))
                W3 = heb_coeffs[idx][15:18].reshape((1,3))
                b1 = heb_coeffs[idx][18:21].reshape((3,1))
                b2 = heb_coeffs[idx][21:24].reshape((3,1))
                b3 = heb_coeffs[idx][24]
                
                X = np.array([o1[i], o2[j]])
                
                weights2_3[:,i][j] += forward_MLP3_withBias(X, W1, W2, W3, b1, b2, b3)
    
    
        heb_offset += weights2_3.shape[1] * weights2_3.shape[0]
        # Layer 3
        for i in range(weights3_4.shape[1]): 
            for j in range(weights3_4.shape[0]):  
                idx = heb_offset + (weights3_4.shape[0]-1)*i + i+j 
                
                W1 = heb_coeffs[idx][:6].reshape((3,2))
                W2 = heb_coeffs[idx][6:15].reshape((3,3))
                W3 = heb_coeffs[idx][15:18].reshape((1,3))
                b1 = heb_coeffs[idx][18:21].reshape((3,1))
                b2 = heb_coeffs[idx][21:24].reshape((3,1))
                b3 = heb_coeffs[idx][24]
                                
                X = np.array([o2[i], o3[j]])

                weights3_4[:,i][j] += forward_MLP3_withBias(X, W1, W2, W3, b1, b2, b3)

        return weights1_2, weights2_3, weights3_4
    
    
@njit
def neural_hebbian_update_ML3(heb_coeffs, weights1_2, weights2_3, weights3_4, o0, o1, o2, o3):
    
        heb_offset = 0
        # Layer 1         
        for i in range(weights1_2.shape[1]): 
            for j in range(weights1_2.shape[0]):  
                idx = (weights1_2.shape[0]-1)*i + i + j
                
                W1 = heb_coeffs[idx][:6].reshape((3,2))
                W2 = heb_coeffs[idx][6:15].reshape((3,3))
                W3 = heb_coeffs[idx][15:18].reshape((1,3))

                X = np.array([o0[i], o1[j]])
                
                weights1_2[:,i][j] += forward_MLP3(X, W1, W2, W3)

                
        heb_offset = weights1_2.shape[1] * weights1_2.shape[0]
        # Layer 2
        for i in range(weights2_3.shape[1]): 
            for j in range(weights2_3.shape[0]):  
                idx = heb_offset + (weights2_3.shape[0]-1)*i + i+j
                
                W1 = heb_coeffs[idx][:6].reshape((3,2))
                W2 = heb_coeffs[idx][6:15].reshape((3,3))
                W3 = heb_coeffs[idx][15:18].reshape((1,3))
                
                X = np.array([o1[i], o2[j]])
                
                weights2_3[:,i][j] += forward_MLP3(X, W1, W2, W3)
    
    
        heb_offset += weights2_3.shape[1] * weights2_3.shape[0]
        # Layer 3
        for i in range(weights3_4.shape[1]): 
            for j in range(weights3_4.shape[0]):  
                idx = heb_offset + (weights3_4.shape[0]-1)*i + i+j 
                
                W1 = heb_coeffs[idx][:6].reshape((3,2))
                W2 = heb_coeffs[idx][6:15].reshape((3,3))
                W3 = heb_coeffs[idx][15:18].reshape((1,3))
                                
                X = np.array([o2[i], o3[j]])

                weights3_4[:,i][j] += forward_MLP3(X, W1, W2, W3)

        return weights1_2, weights2_3, weights3_4
    
    
    
@njit
def neural_hebbian_update_ML2(heb_coeffs, weights1_2, weights2_3, weights3_4, o0, o1, o2, o3):
    
        heb_offset = 0
        # Layer 1         
        for i in range(weights1_2.shape[1]): 
            for j in range(weights1_2.shape[0]):  
                idx = (weights1_2.shape[0]-1)*i + i + j
                
                W1 = heb_coeffs[idx][:6].reshape((3,2))
                W2 = heb_coeffs[idx][6:9].reshape((1,3))

                X = np.array([o0[i], o1[j]])
                
                weights1_2[:,i][j] += forward_MLP2(X, W1, W2)

                
        heb_offset = weights1_2.shape[1] * weights1_2.shape[0]
        # Layer 2
        for i in range(weights2_3.shape[1]): 
            for j in range(weights2_3.shape[0]):  
                idx = heb_offset + (weights2_3.shape[0]-1)*i + i+j
                
                W1 = heb_coeffs[idx][:6].reshape((3,2))
                W2 = heb_coeffs[idx][6:9].reshape((1,3))
                
                X = np.array([o1[i], o2[j]])
                
                weights2_3[:,i][j] += forward_MLP2(X, W1, W2)
    
    
        heb_offset += weights2_3.shape[1] * weights2_3.shape[0]
        # Layer 3
        for i in range(weights3_4.shape[1]): 
            for j in range(weights3_4.shape[0]):  
                idx = heb_offset + (weights3_4.shape[0]-1)*i + i+j 
                
                W1 = heb_coeffs[idx][:6].reshape((3,2))
                W2 = heb_coeffs[idx][6:9].reshape((1,3))
                                
                X = np.array([o2[i], o3[j]])

                weights3_4[:,i][j] += forward_MLP2(X, W1, W2)

        return weights1_2, weights2_3, weights3_4
    
