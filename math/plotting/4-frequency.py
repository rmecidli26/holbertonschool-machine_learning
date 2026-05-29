#!/usr/bin/env python3
"""
Module to plot a histogram using bar function.
"""
import numpy as np
import matplotlib.pyplot as plt


def frequency():
    """
    Plots a histogram using plt.bar to match exact reference requirements.
    """
    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)
    plt.figure(figsize=(6.4, 4.8))

    # Məlumatları 10-luq aralıqlarla qruplaşdırırıq
    counts, bins = np.histogram(student_grades, bins=np.arange(0, 101, 10))

    # Sütunları çəkirik
    plt.bar(bins[:-1], counts, width=10, edgecolor='black', align='edge')

    plt.xlabel('Grades')
    plt.ylabel('Number of Students')
    plt.title('Project A')
    plt.xlim(0, 100)
