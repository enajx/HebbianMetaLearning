import numpy as np
import multiprocessing as mp
import torch
import time
from os.path import exists
from os import mkdir
from gym.spaces import Discrete, Box
import gym
import pybullet_envs 

from fitness_functions import fitness_hebb


def compute_ranks(x):
  """
  Returns rank as a vector of len(x) with integers from 0 to len(x)
  """
  assert x.ndim == 1
  ranks = np.empty(len(x), dtype=int)
  ranks[x.argsort()] = np.arange(len(x))
  return ranks

def compute_centered_ranks(x):
  """
  Maps x to [-0.5, 0.5] and returns the rank
  """
  y = compute_ranks(x.ravel()).reshape(x.shape).astype(np.float32)
  y /= (x.size - 1)
  y -= .5
  return y

            
def worker_process_hebb(arg):
    get_reward_func, hebb_rule,  eng,  init_weights, coeffs = arg
    
    wp = np.array(coeffs)
    decay = - 0.01 * np.mean(wp**2)
    r = get_reward_func( hebb_rule,  eng,  init_weights, coeffs) + decay
    
    return r 


def worker_process_hebb_coevo(arg): 
    get_reward_func,  hebb_rule,  eng,  init_weights, coeffs, coevolved_parameters = arg
    
    wp = np.array(coeffs)
    decay = - 0.01 * np.mean(wp**2)
    r = get_reward_func( hebb_rule,  eng,  init_weights, coeffs, coevolved_parameters) + decay

    return r 


class EvolutionStrategyHebb(object):
    def __init__(self, hebb_rule,  environment, init_weights = 'uni', population_size=100, sigma=0.1, learning_rate=0.2, decay=0.995, num_threads=1, distribution = 'normal'):
        
        self.hebb_rule = hebb_rule                     
        self.environment = environment                         
        self.init_weights = init_weights               
        self.POPULATION_SIZE = population_size
        self.SIGMA = sigma
        self.learning_rate = learning_rate            
        self.decay = decay
        self.num_threads = mp.cpu_count() if num_threads == -1 else num_threads
        self.update_factor = self.learning_rate / (self.POPULATION_SIZE * self.SIGMA)
        self.distribution = distribution                      

        # The number of hebbian coefficients per synapse
        if hebb_rule == 'A':                                                     
            self.coefficients_per_synapse = 1
        elif hebb_rule == 'AD':                                                
            self.coefficients_per_synapse = 2
        elif hebb_rule == 'AD_lr':                                             
            self.coefficients_per_synapse = 3
        elif hebb_rule == 'ABC':                                                
            self.coefficients_per_synapse = 3
        elif hebb_rule == 'ABC_lr':                                             
            self.coefficients_per_synapse = 4
        elif hebb_rule == 'ABCD':                                             
            self.coefficients_per_synapse = 4
        elif hebb_rule == 'ABCD_lr':                                           
            self.coefficients_per_synapse = 5
        elif hebb_rule == 'ABCD_lr_D_out':                                             
            self.coefficients_per_synapse = 5
        elif hebb_rule == 'ABCD_lr_D_in_and_out':                                             
            self.coefficients_per_synapse = 6
        else:
            raise ValueError('The provided Hebbian rule is not valid')
            
       
        # Look up observation and action space dimension
        env = gym.make(environment)    
        if len(env.observation_space.shape) == 3:     # Pixel-based environment
            self.pixel_env = True
        elif len(env.observation_space.shape) == 1:   # State-based environment 
            self.pixel_env = False
            input_dim = env.observation_space.shape[0]
        elif isinstance(env.observation_space, Discrete):
            self.pixel_env = False
            input_dim = env.observation_space.n
        else:
            raise ValueError('Observation space not supported')

        if isinstance(env.action_space, Box):
            action_dim = env.action_space.shape[0]
        elif isinstance(env.action_space, Discrete):
            action_dim = env.action_space.n
        else:
            raise ValueError('Action space not supported')
        
        # Intial weights co-evolution flag:
        self.coevolve_init = True if self.init_weights == 'coevolve' else False
        if self.coevolve_init:
            print('\nCo-evolving initial weights of the network')
       

                    
        # Initialize the values of hebbian coefficients and CNN parameters or initial weights of co-evolving initial weights   
        
        # Pixel-based environments (CNN + MLP)       
        if self.pixel_env:
            cnn_weights = 1362                                                                                        #  CNN: (6, 3, 3, 3) + (8, 6, 5, 5) = 162+1200 = 1362
            plastic_weights = (128*648) + (64*128) + (action_dim*64)                                                  #  Hebbian coefficients: MLP x coefficients_per_synapse : plastic_weights x coefficients_per_synapse
            
            # Co-evolution of initial weights
            if self.coevolve_init:
                if self.distribution == 'uniform':                                                                        
                    self.coeffs = np.random.uniform(-1,1,(plastic_weights, self.coefficients_per_synapse)) 
                    self.initial_weights_co = np.random.uniform(-1,1, (cnn_weights + plastic_weights ,1))  
                    
                elif self.distribution == 'normal':    
                    self.coeffs = torch.randn(plastic_weights, self.coefficients_per_synapse).detach().numpy().squeeze() 
                    self.initial_weights_co = torch.randn(cnn_weights + plastic_weights , 1).detach().numpy().squeeze()                     
            
            # Random initial weights
            else:
                if self.distribution == 'uniform':                                                                        
                    self.coeffs = np.random.uniform(-1,1,(plastic_weights, self.coefficients_per_synapse)) 
                    self.initial_weights_co = np.random.uniform(-1,1,(cnn_weights,1))      
           
                elif self.distribution == 'normal':    
                    self.coeffs = torch.randn(plastic_weights, self.coefficients_per_synapse).detach().numpy().squeeze() 
                    self.initial_weights_co = torch.randn(cnn_weights, 1).detach().numpy().squeeze()    
                
        # State-vector environments (MLP)            
        else:
            plastic_weights = (128*input_dim) + (64*128) + (action_dim*64)                                            #  Hebbian coefficients:  MLP x coefficients_per_synapse :plastic_weights x coefficients_per_synapse
            
            # Co-evolution of initial weights
            if self.coevolve_init:
                if self.distribution == 'uniform': 
                    self.coeffs = np.random.uniform(-1,1,(plastic_weights, self.coefficients_per_synapse))
                    self.initial_weights_co = np.random.uniform(-1,1, (plastic_weights ,1))       
                     
                elif self.distribution == 'normal':
                    self.coeffs = torch.randn(plastic_weights, self.coefficients_per_synapse).detach().numpy().squeeze() 
                    self.initial_weights_co = torch.randn(plastic_weights , 1).detach().numpy().squeeze()                     
            
            # Random initial weights
            else:                                      
                if self.distribution == 'uniform': 
                    self.coeffs = np.random.uniform(-1,1,(plastic_weights, self.coefficients_per_synapse)) 
                elif self.distribution == 'normal':
                    self.coeffs = torch.randn(plastic_weights, self.coefficients_per_synapse).detach().numpy().squeeze() 
                    
                    
                    
        # Load fitness function for the selected environment          
        self.get_reward = fitness_hebb
            
            
    def _get_params_try(self, w, p):

        param_try = []
        for index, i in enumerate(p):
            jittered = self.SIGMA * i
            param_try.append(w[index] + jittered)
        param_try = np.array(param_try).astype(np.float32)
        
        return param_try
        # return w + p*self.SIGMA

    def get_coeffs(self):
        return self.coeffs.astype(np.float32)
    
    def get_coevolved_parameters(self):
        return self.initial_weights_co.astype(np.float32)

    def _get_population(self, coevolved_param = False): 
        
    
        # x_ = np.random.randn(int(self.POPULATION_SIZE/2), self.coeffs.shape[0], self.coeffs[0].shape[0])
        # population = np.concatenate((x_,-1*x_)).astype(np.float32)
        
        population = []
            
        if coevolved_param == False:
            for i in range( int(self.POPULATION_SIZE/2) ):
                x = []
                x2 = []
                for w in self.coeffs:
                    j = np.random.randn(*w.shape)             # j: (coefficients_per_synapse, 1) eg. (5,1)
                    x.append(j)                                                   # x: (coefficients_per_synapse, number of synapses) eg. (92690, 5)
                    x2.append(-j) 
                population.append(x)                                              # population : (population size, coefficients_per_synapse, number of synapses), eg. (10, 92690, 5)
                population.append(x2)
                
        elif coevolved_param == True:
            for i in range( int(self.POPULATION_SIZE/2) ):
                x = []
                x2 = []
                for w in self.initial_weights_co:
                    j = np.random.randn(*w.shape)
                    x.append(j)                    
                    x2.append(-j) 

                population.append(x)               
                population.append(x2)
                
        return np.array(population).astype(np.float32)


    def _get_rewards(self, pool, population):
        if pool is not None:

            worker_args = []
            for p in population:

                heb_coeffs_try1 = []
                for index, i in enumerate(p):
                    jittered = self.SIGMA * i
                    heb_coeffs_try1.append(self.coeffs[index] + jittered) 
                heb_coeffs_try = np.array(heb_coeffs_try1).astype(np.float32)

                worker_args.append( (self.get_reward, self.hebb_rule, self.environment,  self.init_weights,  heb_coeffs_try) )
                
            rewards  = pool.map(worker_process_hebb, worker_args)
            
        else:
            rewards = []
            for p in population:
                heb_coeffs_try = np.array(self._get_params_try(self.coeffs, p))
                rewards.append(self.get_reward( self.hebb_rule, self.environment,  self.init_weights, heb_coeffs_try))
        
        rewards = np.array(rewards).astype(np.float32)
        return rewards
    

    def _get_rewards_coevolved(self, pool, population, population_coevolved):
        if pool is not None:

            worker_args = []
            for z in range(len(population)):

                heb_coeffs_try1 = []
                for index, i in enumerate(population[z]):
                    jittered = self.SIGMA * i
                    heb_coeffs_try1.append(self.coeffs[index] + jittered) 
                heb_coeffs_try = np.array(heb_coeffs_try1).astype(np.float32)
                
                coevolved_parameters_try1 = []
                for index, i in enumerate(population_coevolved[z]):
                    jittered = self.SIGMA * i
                    coevolved_parameters_try1.append(self.initial_weights_co[index] + jittered) 
                coevolved_parameters_try = np.array(coevolved_parameters_try1).astype(np.float32)
            
                worker_args.append( (self.get_reward, self.hebb_rule,  self.environment,  self.init_weights, heb_coeffs_try, coevolved_parameters_try) )
                
            rewards  = pool.map(worker_process_hebb_coevo, worker_args)
            
        else:
            rewards = []
            for z in range(len(population)):
                heb_coeffs_try = np.array(self._get_params_try(self.coeffs, population[z]))
                coevolved_parameters_try = np.array(self._get_params_try(self.initial_weights_co, population_coevolved[z]))
                rewards.append(self.get_reward( self.hebb_rule,  self.environment,  self.init_weights, heb_coeffs_try, coevolved_parameters_try))
        
        rewards = np.array(rewards).astype(np.float32)
        return rewards

    def _update_coeffs(self, rewards, population):
        rewards = compute_centered_ranks(rewards)

        std = rewards.std()
        if std == 0:
            raise ValueError('Variance should not be zero')
                
        rewards = (rewards - rewards.mean()) / std
                
        for index, c in enumerate(self.coeffs):
            layer_population = np.array([p[index] for p in population])
                      
            self.update_factor = self.learning_rate / (self.POPULATION_SIZE * self.SIGMA)                
            self.coeffs[index] = c + self.update_factor * np.dot(layer_population.T, rewards).T 

        if self.learning_rate > 0.001:
            self.learning_rate *= self.decay

        #Decay sigma
        if self.SIGMA>0.01:
            self.SIGMA *= 0.999        
            
        
    def _update_coevolved_param(self, rewards, population):
        rewards = compute_centered_ranks(rewards)

        std = rewards.std()
        if std == 0:
            raise ValueError('Variance should not be zero')
                
        rewards = (rewards - rewards.mean()) / std
                
        for index, w in enumerate(self.initial_weights_co):
            layer_population = np.array([p[index] for p in population])
            
            self.update_factor = self.learning_rate / (self.POPULATION_SIZE * self.SIGMA)                
            self.initial_weights_co[index] = w + self.update_factor * np.dot(layer_population.T, rewards).T



    def run(self, iterations, print_step=10, path='heb_coeffs'):                                                    
        
        id_ = str(int(time.time()))
        if not exists(path + '/' + id_):
            mkdir(path + '/' + id_)
            
        print('Run: ' + id_ + '\n\n........................................................................\n')
            
        pool = mp.Pool(self.num_threads) if self.num_threads > 1 else None
        
        generations_rewards = []

        for iteration in range(iterations):                                                                         # Algorithm 2. Salimans, 2017: https://arxiv.org/abs/1703.03864

            # Evolution of Hebbian coefficients & coevolution of cnn parameters and/or initial weights
            if self.pixel_env or self.coevolve_init:                
                population = self._get_population()                                                                 # Sample normal noise:         Step 5
                population_coevolved = self._get_population(coevolved_param=True)                                   # Sample normal noise:         Step 5
                rewards = self._get_rewards_coevolved(pool, population, population_coevolved)                       # Compute population fitness:  Step 6   
                self._update_coeffs(rewards, population)                                                            # Update coefficients:         Steps 8->12
                self._update_coevolved_param(rewards, population_coevolved)                                         # Update coevolved parameters: Steps 8->12
                
            # Evolution of Hebbian coefficients
            else:
                population = self._get_population()                                                                 # Sample normal noise:         Step 5
                rewards = self._get_rewards(pool, population)                                                       # Compute population fitness:  Step 6
                self._update_coeffs(rewards, population)                                                            # Update coefficients:         Steps 8->12
                
                
            # Print fitness and save Hebbian coefficients and/or Coevolved / CNNs parameters
            if (iteration + 1) % print_step == 0:
                rew_ = rewards.mean()
                print('iter %4i | reward: %3i |  update_factor: %f  lr: %f | sum_coeffs: %i sum_abs_coeffs: %4i' % (iteration + 1, rew_ , self.update_factor, self.learning_rate, int(np.sum(self.coeffs)), int(np.sum(abs(self.coeffs)))), flush=True)
                
                if rew_ > 100:
                    torch.save(self.get_coeffs(),  path + "/"+ id_ + '/HEBcoeffs__' + self.environment + "__rew_" + str(int(rew_)) + '__' + self.hebb_rule + "__init_" + str(self.init_weights) + "__pop_" + str(self.POPULATION_SIZE) + '__coeffs' + "__{}.dat".format(iteration))
                    if self.coevolve_init:
                        torch.save(self.get_coevolved_parameters(),  path + "/"+ id_ + '/HEBcoeffs__' + self.environment + "__rew_" + str(int(rew_)) + '__' + self.hebb_rule + "__init_" + str(self.init_weights) + "__pop_" + str(self.POPULATION_SIZE) + '__coevolved_initial_weights' + "__{}.dat".format(iteration))
                    elif self.pixel_env:
                        torch.save(self.get_coevolved_parameters(),  path + "/"+ id_ + '/HEBcoeffs__' + self.environment + "__rew_" + str(int(rew_)) + '__' + self.hebb_rule + "__init_" + str(self.init_weights) + "__pop_" + str(self.POPULATION_SIZE) + '__CNN_parameters' + "__{}.dat".format(iteration))
                        
                generations_rewards.append(rew_)
                np.save(path + "/"+ id_ + '/Fitness_values_' + id_ + '_' + self.environment + '.npy', np.array(generations_rewards))
       
        if pool is not None:
            pool.close()
            pool.join()