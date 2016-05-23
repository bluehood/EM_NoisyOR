#!/usr/bin/env python
from sys import argv
import numpy as np

input_file = argv[1]
print input_file
data = np.loadtxt(input_file, dtype=int)
np.save(input_file, data)
