"""
Losses module for CycleGAN

This module contains the implementation of the losses used in CycleGAN framework.

The losses are divided into two categories: 
  - Building block losses;
  - Specialized losses.

The building block losses are:
   - L1 Loss, the Least Absolute Deviations
   - L2 Loss, the Least Squared Errors

The specialized losses are:
  - Adversarial losses
  - Cycle consistency losses

"""

import numpy as np

## Building block losses ##

# L1 Loss
def l1_loss(x, y):
  return np.sum(np.abs(x - y))

# L2 Loss
def l2_loss(x, y):
  return np.sum(np.square(x - y))

## Adversarial Losses ##

# TODO

## Cycle Consistency Loss ##
def cycle_consistency_loss(G, F, x, y):
  forward_cycle_consistency = np.mean(l1_loss(F(G(x), x)))
  backward_cycle_consistency = np.mean(l1_loss(G(F(y), y)))
  lambda_value = 10.0
  return lambda_value * (forward_cycle_consistency + backward_cycle_consistency)
