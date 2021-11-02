# Implicit Safe Set Algorithm (ISSA)
Code for paper "**[Model-free Safe Control for Zero-Violation Reinforcement Learning](https://openreview.net/forum?id=UGp6FDaxB0f)**" at Conference on Robot Learning (CoRL) 2021. 

![Comparison_0 5xslow_7mb](https://user-images.githubusercontent.com/52708081/139673561-e79a9f7d-36d7-499d-a375-3a14fb9d85be.gif)

## Running instructions 
- To run he PPO-ISSA algorithm for doing task of reaching goals:
    ```
    python experiment.py --alg ppo_adamba_sc --task mygoal1
    ```
    You can customize:
    - the hazard radius of 0.15 by adding `--hazards_size 0.15`
    - the safety index parameters by adding `--n 1.0 --k 1.0 --sigma 0.04 --threshold 0.0`
    - the robot type by adding `--robot point`
    - the cpu numbers for parallel sampling by adding `--cpu 16`
- To plot learning curves, suppose the log files are stored in the data folder:
    ```
    python plot.py ../data/exp_name1/ ../data/exp_name2/
    ```
## Env setup
- Install Safety Gym
    ```
    cd ISSA/safety_gym/env/safety-gym && pip install -e .
    cd ISSA/safety_gym/alog/safety-starter-agents && pip install -e .
    ```
- Install OpenMPI on ubuntu
    ```
    sudo apt-get install openmpi-bin openmpi-doc libopenmpi-dev
    ```
- Trouble shooting for mpi4py:
    ```
    pip uninstall mpi4py
    pip install mpi4py==3.0.2
    ```
## Reference
- [Safety Gym](https://github.com/openai/safety-gym) 
- [Safety Starter Agents](https://github.com/openai/safety-starter-agents) 






