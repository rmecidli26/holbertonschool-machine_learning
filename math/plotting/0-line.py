#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

def line():
    y = np.arange(0, 11) ** 3
    plt.figure(figsize=(6.4, 4.8))

    # X oxu aralığını 0-dan 10-a qədər təyin edirik
    plt.xlim(0, 10)
    
    # Qrafiki çəkirik: 'r-' 'red' (qırmızı) və solid (düz) xətt deməkdir
    plt.plot(y, 'r-')
