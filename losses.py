"""
Losses module for CycleGAN

This module contains the implementation of the losses used in CycleGAN work.

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
def l1_loss(y, y_hat):
  return np.sum(np.abs(y - y_hat))

# L2 Loss
def l2_loss(y, y_hat):
  return np.sum(np.square(y - y_hat))
