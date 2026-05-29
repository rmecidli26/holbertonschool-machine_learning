#!/usr/bin/env python3
"""
Module to plot a histogram of student grades.
"""
import numpy as np
import matplotlib.pyplot as plt


def frequency():
    """
    Plots a histogram of student grades using plt.bar for exact alignment.
    """
    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)
    plt.figure(figsize=(6.4, 4.8))

    counts, bins = np.histogram(student_grades, bins=np.arange(0, 101, 10))
    plt.bar(bins[:-1], counts, width=10, edgecolor='black', align='edge')

    plt.xlabel('Grades')
    plt.ylabel('Number of Students')
    plt.title('Project A')
    plt.axis([0, 100, 0, 30])
    plt.xticks(np.arange(0, 101, 10))
    plt.yticks(np.arange(0, 31, 5))
