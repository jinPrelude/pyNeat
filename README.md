# pyNeat
Pure Python implementation of [NEAT](http://nn.cs.utexas.edu/keyword?stanley:ec02). 
Although this project is heavily inspired by the official NEAT Python implementation([NEAT-Python](https://github.com/CodeReclaimers/neat-python)), it's different in that it's not intended to be able to graft NEAT to other projects, but to show intuitively how NEAT works. Therefore, much of the code is designed to help code readers intuitively understand.

## installation
### prerequisite
You need following library:
```
> sudo apt install swig libopenmpi-dev # for box2d-py & mpi4py
```
We recommand you to install in anaconda virtual environment to avoid any dependency issues.
```
> git clone https://github.com/jinPrelude/pyNeat.git
> conda create -n pyneat python=3.8
> conda activate pyneat
> cd pyNeat
> pip install -r requirements.txt
```

## Train
You can change the training settings by editing the config file. For instance, you can change NEAT algorithm's crossover ratio for CartPole training simply by changing 'crossover_ratio' variable in cartpole config file :
```
# conf/neat/cartpole.yaml
...
strategy:
  name : neat
  offspring_num: 300
  crossover_ratio: 0.75
...
```
Then, you can use the flag to run training with the desired config settings.
```bash
# training CartPole(POMDP settings by default).
> python run_es.py --cfg-path conf/neat/cartpole.yaml

# training BiedalWalker.
> python run_es.py --cfg-path conf/neat/bipedalwalker.yaml
```
Other running options can be found by running `python run_es.py -h` :
```
optional arguments:
  -h, --help            show this help message and exit
  --cfg-path CFG_PATH   config file to run.
  --seed SEED           random seed.
  --n-workers N_WORKERS
  --generation-num GENERATION_NUM
                        max number of generation iteration.
  --eval-ep-num EVAL_EP_NUM
                        number of model evaluaion per iteration.
  --log                 wandb log
  --save-model-period SAVE_MODEL_PERIOD
                        save model for every n iteration.
```