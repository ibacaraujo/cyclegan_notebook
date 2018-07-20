"""
CycleGAN
"""

import numpy as np
from losses import l1_loss, l2_loss

y = np.array([[1, 1, 1, 1], [1, 1, 1, 1]])
y_hat = np.array([[1, 1, 1, 1], [1, 1, 1, 1]])

print(l1_loss(y, y_hat))
print(l2_loss(y, y_hat))
