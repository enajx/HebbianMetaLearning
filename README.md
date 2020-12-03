 
---

<div align="center">    
 
# Meta-Learning through Hebbian Plasticity in Random Networks   

[![Paper](https://img.shields.io/badge/paper-arxiv.2007.02686-B31B1B.svg)](https://arxiv.org/abs/2007.02686)
[![Conference](http://img.shields.io/badge/NeurIPS-2020-4b44ce.svg)](https://proceedings.neurips.cc//paper/2020/hash/ee23e7ad9b473ad072d57aaa9b2a5222-Abstract.html)

</div>
 
This reposistory contains the code to train Hebbian random networks on any [Gym environment](https://github.com/openai/gym/wiki/Table-of-environments) or [pyBullet environment](https://github.com/bulletphysics/bullet3) as described in our paper [Meta-Learning through Hebbian Plasticity in Random Networks, 2020](https://arxiv.org/abs/2007.02686).
Additionally, you can train any custom environment by [registering them.](https://github.com/openai/gym/wiki/Environments)
<!-- 
<p align="center">
  <img src="images/carsmallest.gif" />
</p> -->
![](images/carsmall-min.gif)


## How to run   
<!-- <img src="http://www.sciweavers.org/tex2img.php?eq=%20%5Csqrt%7Bab%7D%20&bc=White&fc=Black&im=tif&fs=12&ff=arev&edit=0" align="center" border="0" alt=" \sqrt{ab} " width="" height="" /> -->
First, install dependencies. Use `Python >= 3.8`:
```bash
# clone project   
git clone https://github.com/enajx/HebbianMetaLearning   

# install dependencies   
cd HebbianMetaLearning 
pip install -r requirements.txt
 ```   
 Next, use `train_hebb.py` to train an agent. You can train any of OpenAI Gym's or pyBullet environments:
 ```bash

# train Hebbian network to solve the racing car
python train_hebb.py --environment CarRacing-v0


# train Hebbian network specifying evolution parameters, eg. 
python train_hebb.py --environment CarRacing-v0 --hebb_rule ABCD_lr --generations 300 --popsize 200 --print_every 1 --init_weights uni --lr 0.2 --sigma 0.1 --decay 0.995 --threads -1 --distribution normal

```

 Use `python train_hebb.py --help` to display all the training options:


 ```

train_hebb.py [--environment] [--hebb_rule] [--popsize] [--lr] [--decay] [--sigma] [--init_weights] [--print_every] [--generations] [--threads] [--folder] [--distribution]

  --environment    Environment: any OpenAI Gym or pyBullet environment may be used
  --hebb_rule      Hebbian rule type: A, AD_lr, ABC, ABC_lr, ABCD, ABCD_lr
  --popsize        Population size.
  --lr             ES learning rate.
  --decay          ES decay.
  --sigma          ES sigma: modulates the amount of noise used to populate each new generation
  --init_weights   The distribution used to sample random weights from at each episode / coevolve mode: uni, normal, coevolve
  --print_every    Print and save every N steps.
  --generations    Number of generations that the ES will run.
  --threads        Number of threads used to run evolution in parallel.
  --folder         folder to store the evolved Hebbian coefficients
  --distribution   Sampling distribution for initialize the Hebbian coefficients: normal, uniform

```

Once trained, use `evaluate_hebb.py` to test the evolved agent:
 ```

python evaluate_hebb.py --environment CarRacing-v0 --hebb_rule ABCD_lr --path_hebb heb_coeffs.dat --path_coev cnn_parameters.dat --init_weights uni 

```

When running on a headless server some environments will require a virtual display to run -eg. CarRacing-v0-, in this case run:
 ```bash

xvfb-run -a -s "-screen 0 1400x900x24 +extension RANDR" -- python train_hebb.py --environment CarRacing-v0

```

## Citation   

If you use the code for academic or commecial use, please cite the associated paper:

```bibtex

@inproceedings{Najarro2020,
	title = {{Meta-Learning through Hebbian Plasticity in Random Networks}},
	author = {Najarro, Elias and Risi, Sebastian},
	booktitle = {Advances in Neural Information Processing Systems},
	year = {2020},
	url = {https://arxiv.org/abs/2007.02686}
}

```   

## Reproduce the paper's results

The *CarRacing-v0* environment results can be reproduced by running `python train_hebb.py --environment CarRacing-v0`. 

The damaged quadruped morphologies can be found in the folder *damaged_bullet_morphologies*. In order to 
reproduce the damaged quadruped results, these new morphologies need to be firstly [registered as custom environments](https://github.com/openai/gym/wiki/Environments) 
and secondly added to the fitness function: simply add a 2-fold loop which returns the average cummulative distance walked of the standard morphology and the damaged one.

All the necessary training parameters are indicated in the paper.

The static networks used as baselines can be reproduced with the code [in this repository](https://github.com/enajx/ES).

If you have any trouble reproducing the paper's results, feel free to open an issue or email us.


## Some notes on training performance

In the paper we have tested the CarRacing-v0 and AntBulletEnv-v0 environments. For both of them we have written custom functions to bound the actions;
the rest of the environments have a simple clipping mechanism to bound their actions. Environments with a continuous action space (ie. *Box*)
may benefit from a continous scaling -rather than clipping- of their action spaces, either with a custom activation function or with 
Gym's RescaleAction wrapper.

Another element that greatly affects performance -if you have bounded computational resources- is the choice of a suitable early stop meachanism such that less CPU cycles are wasted, 
eg. for the CarRacing-v0 environment we use 20 consecutive steps with negative reward as an early stop signal.

Finally, some pixel-based environments would likely benefit from using grayscaling + stacked frames approach rather than feeding the network the three RGB channels as we do in our 
implementation, eg. by using Gym's [Frame stack wrapper](https://github.com/openai/gym/blob/master/gym/wrappers/frame_stack.py#L58) or the [Atari preprocessing wrapper](https://github.com/openai/gym/blob/master/gym/wrappers/atari_preprocessing.py#L12).