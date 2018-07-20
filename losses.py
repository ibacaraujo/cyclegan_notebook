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
