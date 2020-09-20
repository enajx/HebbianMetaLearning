import gym
import torch

import numpy as np
import pybullet_envs
from gym.spaces import Discrete, Box
from gym import wrappers as w
import pickle
import argparse
import sys

from hebbian_weights_update import *
from policies import MLP_heb, CNN_heb
from wrappers import ScaledFloatFrame

gym.logger.set_level(40)


def evaluate_hebb(hebb_rule : str, environment : str, init_weights = 'uni', render = True , *evolved_parameters: [np.array]) -> None:
    """
    Copypasta function from fitness_functions::fitness_hebb
    It adds rendering of the environment and prints the cumulative episodic reward
    """
    
    
    def weights_init(m):
        if isinstance(m, torch.nn.Linear):
            if init_weights == 'xa_uni':  
                torch.nn.init.xavier_uniform(m.weight.data, 0.3)
            elif init_weights == 'sparse':  
                torch.nn.init.sparse_(m.weight.data, 0.8)
            elif init_weights == 'uni':  
                torch.nn.init.uniform_(m.weight.data, -0.1, 0.1)
            elif init_weights == 'normal':  
                torch.nn.init.normal_(m.weight.data, 0, 0.024)
            elif init_weights == 'ka_uni':  
                torch.nn.init.kaiming_uniform_(m.weight.data, 3)
            elif init_weights == 'uni_big':
                torch.nn.init.uniform_(m.weight.data, -1, 1)
            elif init_weights == 'xa_uni_big':
                torch.nn.init.xavier_uniform(m.weight.data)
            elif init_weights == 'default' or init_weights == None:
                pass
            
    # Unpack evolved parameters
    try: 
        hebb_coeffs, initial_weights_co = evolved_parameters
    except: 
        hebb_coeffs = evolved_parameters[0]
            
    # Intial weights co-evolution flag:
    coevolve_init = True if init_weights == 'coevolve' else False

    with torch.no_grad():
                    
        # Load environment
        try: env = gym.make(environment, verbose = 0)
        except: env = gym.make(environment)
                        
        if environment[-12:-6] == 'Bullet' and render:
            env.render()  # bullet envs            

        # Check if selected env is pixel or state-vector 
        if len(env.observation_space.shape) == 3:     # Pixel-based environment
            pixel_env = True
            env = w.ResizeObservation(env, 84)        # Resize and normilise input   
            env = ScaledFloatFrame(env)
            input_channels = 3
        elif len(env.observation_space.shape) == 1:   # State-based environment (only valid for with a 'Box' observational space)
            pixel_env = False
            input_dim = env.observation_space.shape[0]
            
        # Determine action space dimension
        if isinstance(env.action_space, Box):
            action_dim = env.action_space.shape[0]
        elif isinstance(env.action_space, Discrete):
            action_dim = env.action_space.n
        else:
            raise ValueError('Only Box and Discrete action spaces supported')
        
        # Initialise policy network: A simple MLP for state-vector environments and a CNN+MLP for pixel-based environments
        if pixel_env == True: 
            p = CNN_heb(input_channels, action_dim)      
        else:
            p = MLP_heb(input_dim, action_dim)          
        
        # Initialise weights of the policy network with an specific distribution or with the co-evolved weights
        if coevolve_init:
            torch.nn.utils.vector_to_parameters( torch.tensor (initial_weights_co, dtype=torch.float32 ),  p.parameters() )
        else:       
            p.apply(weights_init)
             # Load CNN paramters
            if pixel_env:
                cnn_weights1 = initial_weights_co[:162]
                cnn_weights2 = initial_weights_co[162:]
                list(p.parameters())[0].data = torch.tensor(cnn_weights1.reshape((6,3,3,3))).float()
                list(p.parameters())[1].data = torch.tensor(cnn_weights2.reshape((8,6,5,5))).float()
        p = p.float()
        
        # Unpack network's weights
        if pixel_env:
            weightsCNN1, weightsCNN2, weights1_2, weights2_3, weights3_4 = list(p.parameters())
        else:
            weights1_2, weights2_3, weights3_4 = list(p.parameters())
            
        
        # Convert weights to numpy so we can JIT them with Numba
        weights1_2 = weights1_2.detach().numpy()
        weights2_3 = weights2_3.detach().numpy()
        weights3_4 = weights3_4.detach().numpy()
        
        observation = env.reset() 
        if pixel_env: observation = np.swapaxes(observation,0,2) #(3, 84, 84)       

        # Burnout phase for the bullet quadruped so it starts off from the floor
        if environment == 'AntBulletEnv-v0':
            action = np.zeros(8)
            for _ in range(40):
                __ = env.step(action)        
        
        # normalised_weights = True
        normalised_weights = False if environment[-12:-6] == 'Bullet' else True

        neg_count = 0
        rew_ep = 0
        t = 0
        while True:
            
            o0, o1, o2, o3 = p([observation])
            o0 = o0.numpy()
            o1 = o1.numpy()
            o2 = o2.numpy()
            
            # Adding bounds to the action space
            if environment == 'CarRacing-v0':
                action = np.array([ torch.tanh(o3[0]), torch.sigmoid(o3[1]), torch.sigmoid(o3[2]) ]) 
                o3 = o3.numpy()
            elif environment[-12:-6] == 'Bullet':
                o3 = torch.tanh(o3).numpy()
                action = o3
            else: 
                if isinstance(env.action_space, Box):
                    action = o3.numpy()
                    action = np.clip(action, env.action_space.low, env.action_space.high)                           
                elif isinstance(env.action_space, Discrete):
                    action = np.argmax(o3).numpy()
                o3 = o3.numpy()
            
            # Environment simulation step
            observation, reward, done, info  = env.step(action)  
            if environment == 'AntBulletEnv-v0':
                reward = env.unwrapped.rewards[1] # Distance walked
            rew_ep += reward
            
            # Render
            if environment[-12:-6] != 'Bullet' and render:
                env.render('human') # Gym envs
            
            if pixel_env: observation = np.swapaxes(observation,0,2) #(3, 84, 84)
                            
            # Breaking conditions
            if environment == 'CarRacing-v0':
                neg_count = neg_count+1 if reward < 0.0 else 0
                if (done or neg_count > 20):
                    break
            elif environment[-12:-6] == 'Bullet':
                if t > 200:
                    neg_count = neg_count+1 if reward < 0.0 else 0
                    if (done or neg_count > 30):
                        break
            else:
                if done:
                    break
            t += 1
            
            #### Episodic/Intra-life hebbian update of the weights
            if hebb_rule == 'A': 
                weights1_2, weights2_3, weights3_4 = hebbian_update_A(hebb_coeffs, weights1_2, weights2_3, weights3_4, o0, o1, o2, o3)
            elif hebb_rule == 'AD':
                weights1_2, weights2_3, weights3_4 = hebbian_update_AD(hebb_coeffs, weights1_2, weights2_3, weights3_4, o0, o1, o2, o3)
            elif hebb_rule == 'AD_lr':
                weights1_2, weights2_3, weights3_4 = hebbian_update_AD_lr(hebb_coeffs, weights1_2, weights2_3, weights3_4, o0, o1, o2, o3)
            elif hebb_rule == 'ABC':
                weights1_2, weights2_3, weights3_4 = hebbian_update_ABC(hebb_coeffs, weights1_2, weights2_3, weights3_4, o0, o1, o2, o3)
            elif hebb_rule == 'ABC_lr':
                weights1_2, weights2_3, weights3_4 = hebbian_update_ABC_lr(hebb_coeffs, weights1_2, weights2_3, weights3_4, o0, o1, o2, o3)
            elif hebb_rule == 'ABCD':
                weights1_2, weights2_3, weights3_4 = hebbian_update_ABCD(hebb_coeffs, weights1_2, weights2_3, weights3_4, o0, o1, o2, o3)
            elif hebb_rule == 'ABCD_lr':
                weights1_2, weights2_3, weights3_4 = hebbian_update_ABCD_lr_D_in(hebb_coeffs, weights1_2, weights2_3, weights3_4, o0, o1, o2, o3)
            elif hebb_rule == 'ABCD_lr_D_out':
                weights1_2, weights2_3, weights3_4 = hebbian_update_ABCD_lr_D_out(hebb_coeffs, weights1_2, weights2_3, weights3_4, o0, o1, o2, o3)
            elif hebb_rule == 'ABCD_lr_D_in_and_out':
                weights1_2, weights2_3, weights3_4 = hebbian_update_ABCD_lr_D_in_and_out(hebb_coeffs, weights1_2, weights2_3, weights3_4, o0, o1, o2, o3)
            else:
                raise ValueError('The provided Hebbian rule is not valid')


            # Normalise weights per layer
            if normalised_weights == True:
                (a, b, c) = (0, 1, 2) if not pixel_env else (2, 3, 4)
                list(p.parameters())[a].data /= list(p.parameters())[a].__abs__().max()
                list(p.parameters())[b].data /= list(p.parameters())[b].__abs__().max()
                list(p.parameters())[c].data /= list(p.parameters())[c].__abs__().max()
            
        env.close()
        
        print('\n Episode cumulative rewards ', int(rew_ep))

    
    
    
def main(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--environment', type=str, default='CarRacing-v0', metavar='', help='Gym environment: any OpenAI Gym may be used')
    parser.add_argument('--hebb_rule', type=str,  default = 'ABCD_lr', metavar='', help='Hebbian rule type: A, AD, AD_lr, ABC, ABC_lr, ABCD, ABCD_lr, ABCD_lr_D_out, ABCD_lr_D_in_and_out')    
    parser.add_argument('--init_weights', type=str,  default = 'uni', metavar='', help='Weight initilisation distribution used to sample from at each episode: uni, normal, default, xa_uni, sparse, ka_uni')
    parser.add_argument('--path_hebb', type=str,  default = None, metavar='', help='path to the evolved Hebbian coefficients')
    parser.add_argument('--path_coev', type=str,  default = None, metavar='', help='path to the evolved CNN parameters or the coevolve initial weights')

    args = parser.parse_args()

    hebb_coeffs = torch.load(args.path_hebb)
    coevolved_or_cnn_parameters = torch.load(args.path_coev) if args.path_coev is not None else None    
    init_weights = 'uni' 
    render = True
    
    # Run the environment
    evaluate_hebb(args.hebb_rule, args.environment, args.init_weights, render, hebb_coeffs, coevolved_or_cnn_parameters)
    
if __name__ == '__main__':
    main(sys.argv)

    
    
