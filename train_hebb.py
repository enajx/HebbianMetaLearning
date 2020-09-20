import time
import argparse
import sys
import torch
from os.path import join, exists
from os import mkdir
import gym

from evolution_strategy_hebb import EvolutionStrategyHebb

torch.set_num_threads(1)
gym.logger.set_level(40)


def main(argv):
    parser = argparse.ArgumentParser()
      
    parser.add_argument('--environment', type=str, default='CarRacing-v0', metavar='', help='Environment: any OpenAI Gym or pyBullet environment may be used')
    parser.add_argument('--hebb_rule', type=str,  default = 'ABCD_lr', metavar='', help='Hebbian rule type: A, AD, AD_lr, ABC, ABC_lr, ABCD, ABCD_lr, ABCD_lr_D_out, ABCD_lr_D_in_and_out')
    parser.add_argument('--popsize', type=int,  default = 200, metavar='', help='Population size.') 
    parser.add_argument('--lr', type=float,  default = 0.2, metavar='', help='ES learning rate.') 
    parser.add_argument('--decay', type=float,  default = 0.995, metavar='', help='ES learning rate decay.')  
    parser.add_argument('--sigma', type=float,  default = 0.1, metavar='', help='ES sigma: modulates the amount of noise used to populate each new generation') 
    parser.add_argument('--init_weights', type=str,  default = 'uni', metavar='', help='The distribution used to sample random weights from at each episode: uni, normal, default, xa_uni, sparse, ka_uni or coevolve to co-evolve the intial weights')
    parser.add_argument('--print_every', type=int, default = 1, metavar='', help='Print and save every N steps.') 
    parser.add_argument('--generations', type=int, default= 300, metavar='', help='Number of generations that the ES will run.')
    parser.add_argument('--threads', type=int, metavar='', default = -1, help='Number of threads used to run evolution in parallel: -1 uses all threads available')    
    parser.add_argument('--folder', type=str, default='heb_coeffs', metavar='', help='folder to store the evolved Hebbian coefficients')
    parser.add_argument('--distribution', type=str, default='normal', metavar='', help='Sampling distribution for initialize the Hebbian coefficients: normal, uniform')

    args = parser.parse_args()

    if not exists(args.folder):
        mkdir(args.folder)
        
    # Initialise the EvolutionStrategy class
    print('\n\n........................................................................')
    print('\nInitilisating Hebbian ES for ' + args.environment + ' with ' + args.hebb_rule + ' Hebbian rule\n')
    es = EvolutionStrategyHebb(args.hebb_rule, args.environment, args.init_weights, population_size=args.popsize, sigma=args.sigma, learning_rate=args.lr, decay=args.decay, num_threads=args.threads, distribution=args.distribution)
    
    # Start the evolution
    print('\n........................................................................')
    print('\n ♪┏(°.°)┛┗(°.°)┓ Starting Evolution ┗(°.°)┛┏(°.°)┓ ♪ \n')
    tic = time.time()
    es.run(args.generations, print_step=args.print_every, path=args.folder)
    toc = time.time()
    print('\nEvolution took: ', int(toc-tic), ' seconds\n')
    
    
if __name__ == '__main__':
    main(sys.argv)


