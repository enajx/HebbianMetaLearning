import numpy as np
from numba import njit

    
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
    
 