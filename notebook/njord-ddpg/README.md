# Njord-ddpg

Implementation of the DeepMind's Deep Deterministic Policy Gradient (DDPG)algorithm applied to crypto-currency trading. 

## Problem statement

The aim of this project is to train an algorithm to trade in an autonomous fashion. The DDPG algorithm allows us to take continuous actions. Since buying or selling action are discrete, we have to find a way to express them in a continuous form. For this purpose, we can train the algorithm to take either long or short positions. The problem formulation is actually taken from [this paper](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.1.7210&rep=rep1&type=pdf). Therefore, the algorithm outputs an action between -1 and +1.

## Dataset

The dataset used for training is provided by us. The sampling frequency is equal to 10 minutes or 600 seconds. The dataset is built based on prices recorded every 5 seconds and aggregated trades on the binance exchange platform for the BTC/USDT pair. The dataset coveres the period from 2018-10-01 to 2018-10-14.

The dataset has the following set of features.

    1 Open price
    2 High price
    3 Low price
    4 Close price
    5 Avg price
    6 Buy and Sell volume imbalance

## Credits

The code implementing the DDPG algorithm was originaly taken from [here](https://github.com/vy007vikas/PyTorch-ActorCriticRL). Many thanks!

## Disclaimer

This code is not intended for trading. Use it at your own risk if you want too.

## References

* [DDPG paper by DeepMind](https://arxiv.org/abs/1509.02971)
* [DDPG blog by penami4911](http://pemami4911.github.io/blog/2016/08/21/ddpg-rl.html)
* [Reinforcement Learning for trading](https://papers.nips.cc/paper/1551-reinforcement-learning-for-trading.pdf)
* [Learning to Trade via Direct Reinforcement](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.1.7210&rep=rep1&type=pdf)

-----------

Copyright (C) 2018
