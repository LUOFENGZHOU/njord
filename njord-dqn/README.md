# Njord-dqn

Implementation of the DeepMind's Deep Q-Network algorithm applied to crypto-currency trading. 

## Problem statement

The aim of this project is to train an agent to trade in an autonomous way. The agent should be able to consider when he should buy or sell the coins and just wait. He then gets rewarded for taking actions leading to improvement of his portfolio. The DQN algorithm allows us to take discrete actions. The agent could for instance take the following actions:
    1. Buy
    2. Sell
    3. Do nothing

We decide to consider the problem a slightly different manner. Instead of considering explicitly the actions of buying or selling, one can think of the agent taking a long or short position, ie:
    1. Long.
    2. Short.

The actions are then taken implicitly. Switching from one position to the other triggers the buy or sell trade. The buy trades are triggered when switching from a short to a long postion. Inversely, the sell actions are triggered when switching from a long to a short position.

## Dataset

The dataset used for training is provided by us. The sampling frequency is equal to 10 minutes (600 seconds). The dataset is built based on prices recorded every 5 seconds and aggregated trades on the binance exchange platform for the BTC/USDT pair. The dataset coveres the period from 2018-10-01 to 2018-10-14.

The dataset has the following set of features.

    1 Open price
    2 High price
    3 Low price
    4 Close price
    5 Avg price
    6 Buy and Sell volume imbalance

## Improvements

1. The epsilon greedy exploration scheme might not be suited for the exploration of the environment regarding autonomous trading strategies. Other exploration strategies should be investigated. 

## Disclaimer

This code is not intended for trading. Use it at your own risk if you want too.

## References

* [DQN paper by DeepMind](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)
* [Reinforcement Learning for trading](https://papers.nips.cc/paper/1551-reinforcement-learning-for-trading.pdf)
* [Learning to Trade via Direct Reinforcement](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.1.7210&rep=rep1&type=pdf)

-----------

Copyright (C) 2018
