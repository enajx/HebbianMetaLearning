import numpy as np
from numba import njit

import torch
import torch.nn as nn

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# class neuralHebb(nn.Module):
#     "MLP, no bias"
#     def __init__(self, input_space=2, action_space=1, bias=True):
#         super(neuralHebb, self).__init__()

#         self.fc1 = nn.Linear(input_space, 3, bias=bias)
#         self.fc2 = nn.Linear(3, 3, bias=bias)
#         self.fc3 = nn.Linear(3, action_space, bias=bias)

#     def forward(self, ob):
#         state = torch.as_tensor(ob).float().detach()
        
#         x1 = torch.tanh(self.fc1(state))   
#         x2 = torch.tanh(self.fc2(x1))
#         o = self.fc3(x2)  
         
#         return o
    
# @njit
# class neuralHebb:
#     def __init__(self, n_in=2, n_hidden=3, n_out=1):
#         # Network dimensions
#         self.n_x = n_in
#         self.n_h = n_hidden
#         self.n_y = n_out

#         # Parameters initialization
#         self.W1 = np.random.randn(self.n_h, self.n_x) # (3, 2)
#         # self.b1 = np.zeros((self.n_h, 1)) # (3, 1)
#         self.W2 = np.random.randn(self.n_y, self.n_h) # (1, 3)
#         # self.b2 = np.zeros((self.n_y, 1)) # (1, 1)

#     def __call__(self, X):
#         """ Forward computation """
#         self.Z1 = self.W1.dot(X.T) 
#         self.A1 = np.tanh(self.Z1)
#         self.Z2 = self.W2.dot(self.A1)
#         self.A2 = np.tanh(self.Z2)
#         # self.Z1 = self.W1.dot(X.T) + self.b1
#         # self.A1 = np.tanh(self.Z1)
#         # self.Z2 = self.W2.dot(self.A1) + self.b2
#         # self.A2 = np.tanh(self.Z2)
        
#         return self.A2

# def forward(X, W1, W2):
    
#     Z1 = W1.dot(X.T) 
#     A1 = np.tanh(Z1)
#     Z2 = W2.dot(A1)
#     A2 = np.tanh(Z2)

#     return A2

    
@njit
def neural_hebbian_update(heb_coeffs, weights1_2, weights2_3, weights3_4, o0, o1, o2, o3):
    
               
        heb_offset = 0
        # Layer 1         
        for i in range(weights1_2.shape[1]): 
            for j in range(weights1_2.shape[0]):  
                idx = (weights1_2.shape[0]-1)*i + i + j
                
                # W1 = heb_coeffs[idx][:6].reshape((3,2))
                # W2 = heb_coeffs[idx][6:9].reshape((1,3))
                
                W1 = heb_coeffs[idx][:6].reshape((3,2))
                W2 = heb_coeffs[idx][6:15].reshape((3,3))
                W3 = heb_coeffs[idx][15:18].reshape((1,3))
                
                # b1 = heb_coeffs[idx][10:13].reshape((3,1))
                # b2 = heb_coeffs[idx][14]
                # b2 = heb_coeffs[idx][14].reshape((1,1))
                
                X = np.array([o0[i], o1[j]])
                
                Z1 = W1.dot(X.T) 
                A1 = np.tanh(Z1)
                Z2 = W2.dot(A1) 
                A2 = np.tanh(Z2)
                Z3 = W3.dot(A2) 
                delta = np.tanh(Z3)

                weights1_2[:,i][j] += delta.item()
                # weights2_3[:,i][j] += forward(np.array([o0[i], o1[j]]), heb_coeffs[idx][:6].reshape((3,2)), heb_coeffs[idx][6:9].reshape((1,3)))

                
        heb_offset = weights1_2.shape[1] * weights1_2.shape[0]
        # Layer 2
        for i in range(weights2_3.shape[1]): 
            for j in range(weights2_3.shape[0]):  
                idx = heb_offset + (weights2_3.shape[0]-1)*i + i+j
                
                # W1 = heb_coeffs[idx][:6].reshape((3,2))
                # W2 = heb_coeffs[idx][6:9].reshape((1,3))
                
                W1 = heb_coeffs[idx][:6].reshape((3,2))
                W2 = heb_coeffs[idx][6:15].reshape((3,3))
                W3 = heb_coeffs[idx][15:18].reshape((1,3))
                
                X = np.array([o1[i], o2[j]])
                
                Z1 = W1.dot(X.T) 
                A1 = np.tanh(Z1)
                Z2 = W2.dot(A1) 
                A2 = np.tanh(Z2)
                Z3 = W3.dot(A2) 
                delta = np.tanh(Z3)

                weights2_3[:,i][j] += delta.item()
                # weights2_3[:,i][j] += forward(np.array([o1[i], o2[j]]), heb_coeffs[idx][:6].reshape((3,2)), heb_coeffs[idx][6:9].reshape((1,3)))
    
    
        heb_offset += weights2_3.shape[1] * weights2_3.shape[0]
        # Layer 3
        for i in range(weights3_4.shape[1]): 
            for j in range(weights3_4.shape[0]):  
                idx = heb_offset + (weights3_4.shape[0]-1)*i + i+j 
                
                # W1 = heb_coeffs[idx][:6].reshape((3,2))
                # W2 = heb_coeffs[idx][6:9].reshape((1,3))

                W1 = heb_coeffs[idx][:6].reshape((3,2))
                W2 = heb_coeffs[idx][6:15].reshape((3,3))
                W3 = heb_coeffs[idx][15:18].reshape((1,3))
                
                X = np.array([o2[i], o3[j]])

                Z1 = W1.dot(X.T) 
                A1 = np.tanh(Z1)
                Z2 = W2.dot(A1) 
                A2 = np.tanh(Z2)
                Z3 = W3.dot(A2) 
                delta = np.tanh(Z3)

                weights3_4[:,i][j] += delta.item()
                # weights3_4[:,i][j] += forward(np.array([o2[i], o3[j]]), heb_coeffs[idx][:6].reshape((3,2)), heb_coeffs[idx][6:9].reshape((1,3)))

        return weights1_2, weights2_3, weights3_4
    
# def neural_hebbian_update(heb_coeffs, weights1_2, weights2_3, weights3_4, o0, o1, o2, o3):
        
#         hebbiannet = neuralHebb()
        
#         heb_offset = 0
#         # Layer 1         
#         for i in range(weights1_2.shape[1]): 
#             for j in range(weights1_2.shape[0]):  
#                 idx = (weights1_2.shape[0]-1)*i + i + j
#                 nn.utils.vector_to_parameters( torch.tensor (heb_coeffs[idx], dtype=torch.float32 ),  hebbiannet.parameters() )
#                 weights1_2[:,i][j] += hebbiannet(np.array([o0[i], o1[j]]))

#         heb_offset = weights1_2.shape[1] * weights1_2.shape[0]
#         # Layer 2
#         for i in range(weights2_3.shape[1]): 
#             for j in range(weights2_3.shape[0]):  
#                 idx = heb_offset + (weights2_3.shape[0]-1)*i + i+j
#                 nn.utils.vector_to_parameters( torch.tensor (heb_coeffs[idx], dtype=torch.float32 ),  hebbiannet.parameters() )
#                 weights2_3[:,i][j] += hebbiannet(np.array([o1[i], o2[j]]))
    
    
#         heb_offset += weights2_3.shape[1] * weights2_3.shape[0]
#         # Layer 3
#         for i in range(weights3_4.shape[1]): 
#             for j in range(weights3_4.shape[0]):  
#                 idx = heb_offset + (weights3_4.shape[0]-1)*i + i+j 
#                 nn.utils.vector_to_parameters( torch.tensor (heb_coeffs[idx], dtype=torch.float32 ),  hebbiannet.parameters() )
#                 weights3_4[:,i][j] += hebbiannet(np.array([o2[i], o3[j]]))

#         return weights1_2, weights2_3, weights3_4
    
@njit
def hebbian_update_A(heb_coeffs, weights1_2, weights2_3, weights3_4, o0, o1, o2, o3):
        
        heb_offset = 0
        # Layer 1         
        for i in range(weights1_2.shape[1]): 
            for j in range(weights1_2.shape[0]):  
                idx = (weights1_2.shape[0]-1)*i + i + j
                weights1_2[:,i][j] += heb_coeffs[idx] * o0[i] * o1[j]  

        heb_offset = weights1_2.shape[1] * weights1_2.shape[0]
        # Layer 2
        for i in range(weights2_3.shape[1]): 
            for j in range(weights2_3.shape[0]):  
                idx = heb_offset + (weights2_3.shape[0]-1)*i + i+j
                weights2_3[:,i][j] += heb_coeffs[idx] * o1[i] * o2[j] 
    
    
        heb_offset += weights2_3.shape[1] * weights2_3.shape[0]
        # Layer 3
        for i in range(weights3_4.shape[1]): 
            for j in range(weights3_4.shape[0]):  
                idx = heb_offset + (weights3_4.shape[0]-1)*i + i+j 
                weights3_4[:,i][j] += heb_coeffs[idx] * o2[i] * o3[j] 

        return weights1_2, weights2_3, weights3_4
    


@njit
def hebbian_update_AD(heb_coeffs, weights1_2, weights2_3, weights3_4, o0, o1, o2, o3):
        
        heb_offset = 0
        # Layer 1         
        for i in range(weights1_2.shape[1]): 
            for j in range(weights1_2.shape[0]):  
                idx = (weights1_2.shape[0]-1)*i + i + j
                weights1_2[:,i][j] += heb_coeffs[idx][0] * o0[i] * o1[j] + heb_coeffs[idx][1] 

        heb_offset = weights1_2.shape[1] * weights1_2.shape[0]
        # Layer 2
        for i in range(weights2_3.shape[1]): 
            for j in range(weights2_3.shape[0]):  
                idx = heb_offset + (weights2_3.shape[0]-1)*i + i+j
                weights2_3[:,i][j] += heb_coeffs[idx][0] * o1[i] * o2[j] + heb_coeffs[idx][1]  
    
    
        heb_offset += weights2_3.shape[1] * weights2_3.shape[0]
        # Layer 3
        for i in range(weights3_4.shape[1]): 
            for j in range(weights3_4.shape[0]):  
                idx = heb_offset + (weights3_4.shape[0]-1)*i + i+j 
                weights3_4[:,i][j] += heb_coeffs[idx][0] * o2[i] * o3[j] + heb_coeffs[idx][1] 


        return weights1_2, weights2_3, weights3_4
    
@njit
def hebbian_update_AD_lr(heb_coeffs, weights1_2, weights2_3, weights3_4, o0, o1, o2, o3):
        
        heb_offset = 0
        # Layer 1         
        for i in range(weights1_2.shape[1]): 
            for j in range(weights1_2.shape[0]):  
                idx = (weights1_2.shape[0]-1)*i + i + j
                weights1_2[:,i][j] += (heb_coeffs[idx][0] * o0[i] * o1[j] + heb_coeffs[idx][1]) *  heb_coeffs[idx][2] 

        heb_offset = weights1_2.shape[1] * weights1_2.shape[0]
        # Layer 2
        for i in range(weights2_3.shape[1]): 
            for j in range(weights2_3.shape[0]):  
                idx = heb_offset + (weights2_3.shape[0]-1)*i + i+j
                weights2_3[:,i][j] += (heb_coeffs[idx][0] * o1[i] * o2[j] + heb_coeffs[idx][1]) *  heb_coeffs[idx][2]   
    
    
        heb_offset += weights2_3.shape[1] * weights2_3.shape[0]
        # Layer 3
        for i in range(weights3_4.shape[1]): 
            for j in range(weights3_4.shape[0]):  
                idx = heb_offset + (weights3_4.shape[0]-1)*i + i+j 
                weights3_4[:,i][j] += (heb_coeffs[idx][0] * o2[i] * o3[j] + heb_coeffs[idx][1]) *  heb_coeffs[idx][2] 


        return weights1_2, weights2_3, weights3_4



@njit
def hebbian_update_ABC(heb_coeffs, weights1_2, weights2_3, weights3_4, o0, o1, o2, o3):
        
        heb_offset = 0
        # Layer 1         
        for i in range(weights1_2.shape[1]): 
            for j in range(weights1_2.shape[0]):  
                idx = (weights1_2.shape[0]-1)*i + i + j
                weights1_2[:,i][j] += ( heb_coeffs[idx][0] * o0[i] * o1[j]
                                      + heb_coeffs[idx][1] * o0[i] 
                                      + heb_coeffs[idx][2]         * o1[j])  

        heb_offset += weights1_2.shape[1] * weights1_2.shape[0]
        # Layer 2
        for i in range(weights2_3.shape[1]): 
            for j in range(weights2_3.shape[0]):  
                idx = heb_offset + (weights2_3.shape[0]-1)*i + i+j
                weights2_3[:,i][j] += ( heb_coeffs[idx][0] * o1[i] * o2[j]
                                      + heb_coeffs[idx][1] * o1[i] 
                                      + heb_coeffs[idx][2]         * o2[j])  
    
        heb_offset += weights2_3.shape[1] * weights2_3.shape[0]
        # Layer 3
        for i in range(weights3_4.shape[1]): 
            for j in range(weights3_4.shape[0]):  
                idx = heb_offset + (weights3_4.shape[0]-1)*i + i+j 
                weights3_4[:,i][j] += ( heb_coeffs[idx][0] * o2[i] * o3[j]
                                      + heb_coeffs[idx][1] * o2[i] 
                                      + heb_coeffs[idx][2]         * o3[j])  

        return weights1_2, weights2_3, weights3_4


@njit
def hebbian_update_ABC_lr(heb_coeffs, weights1_2, weights2_3, weights3_4, o0, o1, o2, o3):
        
        heb_offset = 0
        # Layer 1        
        for i in range(weights1_2.shape[1]): 
            for j in range(weights1_2.shape[0]):  
                idx = (weights1_2.shape[0]-1)*i + i + j
                weights1_2[:,i][j] += heb_coeffs[idx][3] * ( heb_coeffs[idx][0] * o0[i] * o1[j]
                                                           + heb_coeffs[idx][1] * o0[i] 
                                                           + heb_coeffs[idx][2]         * o1[j])

        heb_offset += weights1_2.shape[1] * weights1_2.shape[0]
        # Layer 2
        for i in range(weights2_3.shape[1]): 
            for j in range(weights2_3.shape[0]):  
                idx = heb_offset + (weights2_3.shape[0]-1)*i + i+j
                weights2_3[:,i][j] += heb_coeffs[idx][3] * ( heb_coeffs[idx][0] * o1[i] * o2[j]
                                                           + heb_coeffs[idx][1] * o1[i] 
                                                           + heb_coeffs[idx][2]         * o2[j])  
    
        heb_offset += weights2_3.shape[1] * weights2_3.shape[0]
        # Layer 3
        for i in range(weights3_4.shape[1]): 
            for j in range(weights3_4.shape[0]):  
                idx = heb_offset + (weights3_4.shape[0]-1)*i + i+j 
                weights3_4[:,i][j] += heb_coeffs[idx][3] * ( heb_coeffs[idx][0] * o2[i] * o3[j]
                                                           + heb_coeffs[idx][1] * o2[i] 
                                                           + heb_coeffs[idx][2]         * o3[j])  

        return weights1_2, weights2_3, weights3_4

@njit
def hebbian_update_ABCD(heb_coeffs, weights1_2, weights2_3, weights3_4, o0, o1, o2, o3):
        
        heb_offset = 0
        # Layer 1        
        for i in range(weights1_2.shape[1]): 
            for j in range(weights1_2.shape[0]):  
                idx = (weights1_2.shape[0]-1)*i + i + j
                weights1_2[:,i][j] += heb_coeffs[idx][3] + ( heb_coeffs[idx][0] * o0[i] * o1[j]
                                                           + heb_coeffs[idx][1] * o0[i] 
                                                           + heb_coeffs[idx][2]         * o1[j])

        heb_offset += weights1_2.shape[1] * weights1_2.shape[0]
        # Layer 2
        for i in range(weights2_3.shape[1]): 
            for j in range(weights2_3.shape[0]):  
                idx = heb_offset + (weights2_3.shape[0]-1)*i + i+j
                weights2_3[:,i][j] += heb_coeffs[idx][3] + ( heb_coeffs[idx][0] * o1[i] * o2[j]
                                                           + heb_coeffs[idx][1] * o1[i] 
                                                           + heb_coeffs[idx][2]         * o2[j])  
    
        heb_offset += weights2_3.shape[1] * weights2_3.shape[0]
        # Layer 3
        for i in range(weights3_4.shape[1]): 
            for j in range(weights3_4.shape[0]):  
                idx = heb_offset + (weights3_4.shape[0]-1)*i + i+j 
                weights3_4[:,i][j] += heb_coeffs[idx][3] + ( heb_coeffs[idx][0] * o2[i] * o3[j]
                                                           + heb_coeffs[idx][1] * o2[i] 
                                                           + heb_coeffs[idx][2]         * o3[j])  
                
        return weights1_2, weights2_3, weights3_4
    
    
@njit    
def hebbian_update_ABCD_lr_D_in(heb_coeffs, weights1_2, weights2_3, weights3_4, o0, o1, o2, o3):
       
        heb_offset = 0
        ## Layer 1         
        for i in range(weights1_2.shape[1]): 
            for j in range(weights1_2.shape[0]): 
                idx = (weights1_2.shape[0]-1)*i + i + j
                weights1_2[:,i][j] += heb_coeffs[idx][3] * ( heb_coeffs[idx][0] * o0[i] * o1[j]
                                                           + heb_coeffs[idx][1] * o0[i] 
                                                           + heb_coeffs[idx][2]         * o1[j]  + heb_coeffs[idx][4])

        heb_offset += weights1_2.shape[1] * weights1_2.shape[0]
        # Layer 2
        for i in range(weights2_3.shape[1]): 
            for j in range(weights2_3.shape[0]):
                idx = heb_offset + (weights2_3.shape[0]-1)*i + i+j
                weights2_3[:,i][j] += heb_coeffs[idx][3] * ( heb_coeffs[idx][0] * o1[i] * o2[j]
                                                           + heb_coeffs[idx][1] * o1[i] 
                                                           + heb_coeffs[idx][2]         * o2[j]  + heb_coeffs[idx][4])

        heb_offset += weights2_3.shape[1] * weights2_3.shape[0]
        # Layer 3
        for i in range(weights3_4.shape[1]): 
            for j in range(weights3_4.shape[0]): 
                idx = heb_offset + (weights3_4.shape[0]-1)*i + i+j 
                weights3_4[:,i][j] += heb_coeffs[idx][3] * ( heb_coeffs[idx][0] * o2[i] * o3[j]
                                                           + heb_coeffs[idx][1] * o2[i] 
                                                           + heb_coeffs[idx][2]         * o3[j]  + heb_coeffs[idx][4])
                
        return weights1_2, weights2_3, weights3_4
    
    
@njit
def hebbian_update_ABCD_lr_D_out(heb_coeffs, weights1_2, weights2_3, weights3_4, o0, o1, o2, o3):
       
        heb_offset = 0
        # Layer 1         
        for i in range(weights1_2.shape[1]): 
            for j in range(weights1_2.shape[0]):  
                idx = (weights1_2.shape[0]-1)*i + i + j
                weights1_2[:,i][j] += heb_coeffs[idx][3] * ( heb_coeffs[idx][0] * o0[i] * o1[j]
                                                           + heb_coeffs[idx][1] * o0[i] 
                                                           + heb_coeffs[idx][2]         * o1[j])  + heb_coeffs[idx][4]

        heb_offset += weights1_2.shape[1] * weights1_2.shape[0]
        # Layer 2
        for i in range(weights2_3.shape[1]): 
            for j in range(weights2_3.shape[0]):  
                idx = heb_offset + (weights2_3.shape[0]-1)*i + i+j
                weights2_3[:,i][j] += heb_coeffs[idx][3] * ( heb_coeffs[idx][0] * o1[i] * o2[j]
                                                           + heb_coeffs[idx][1] * o1[i] 
                                                           + heb_coeffs[idx][2]         * o2[j])  + heb_coeffs[idx][4]
    
        heb_offset += weights2_3.shape[1] * weights2_3.shape[0]
        # Layer 3
        for i in range(weights3_4.shape[1]): 
            for j in range(weights3_4.shape[0]):  
                idx = heb_offset + (weights3_4.shape[0]-1)*i + i+j 
                weights3_4[:,i][j] += heb_coeffs[idx][3] * ( heb_coeffs[idx][0] * o2[i] * o3[j]
                                                           + heb_coeffs[idx][1] * o2[i] 
                                                           + heb_coeffs[idx][2]         * o3[j])  + heb_coeffs[idx][4]

        return weights1_2, weights2_3, weights3_4

@njit
def hebbian_update_ABCD_lr_D_in_and_out(heb_coeffs, weights1_2, weights2_3, weights3_4, o0, o1, o2, o3):
       
        heb_offset = 0
        # Layer 1         
        for i in range(weights1_2.shape[1]): 
            for j in range(weights1_2.shape[0]):  
                idx = (weights1_2.shape[0]-1)*i + i + j
                weights1_2[:,i][j] += heb_coeffs[idx][3] * ( heb_coeffs[idx][0] * o0[i] * o1[j]
                                                           + heb_coeffs[idx][1] * o0[i] 
                                                           + heb_coeffs[idx][2]         * o1[j]  + heb_coeffs[idx][4]) + heb_coeffs[idx][5]

        heb_offset += weights1_2.shape[1] * weights1_2.shape[0]
        # Layer 2
        for i in range(weights2_3.shape[1]): 
            for j in range(weights2_3.shape[0]):  
                idx = heb_offset + (weights2_3.shape[0]-1)*i + i+j
                weights2_3[:,i][j] += heb_coeffs[idx][3] * ( heb_coeffs[idx][0] * o1[i] * o2[j]
                                                           + heb_coeffs[idx][1] * o1[i] 
                                                           + heb_coeffs[idx][2]         * o2[j]  + heb_coeffs[idx][4]) + heb_coeffs[idx][5]
    
        heb_offset += weights2_3.shape[1] * weights2_3.shape[0]
        # Layer 3
        for i in range(weights3_4.shape[1]): 
            for j in range(weights3_4.shape[0]):  
                idx = heb_offset + (weights3_4.shape[0]-1)*i + i+j 
                weights3_4[:,i][j] += heb_coeffs[idx][3] * ( heb_coeffs[idx][0] * o2[i] * o3[j]
                                                           + heb_coeffs[idx][1] * o2[i] 
                                                           + heb_coeffs[idx][2]         * o3[j]  + heb_coeffs[idx][4]) + heb_coeffs[idx][5]

        return weights1_2, weights2_3, weights3_4
    
 