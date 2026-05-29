#!/usr/bin/env python3
"""
Module to plot a stacked bar graph of fruit per person.
"""
import numpy as np
import matplotlib.pyplot as plt


def bars():
    """
    Plots a stacked bar graph with specified colors and layout.
    """
    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4, 3))
    plt.figure(figsize=(6.4, 4.8))

    people = ['Farrah', 'Fred', 'Felicia']
    fruits = ['apples', 'bananas', 'oranges', 'peaches']
    colors = ['red', 'yellow', '#ff8000', '#ffe5b4']

    bottom = np.zeros(3)

    for i in range(len(fruit)):
        plt.bar(people, fruit[i], width=0.5, bottom=bottom,
                label=fruits[i], color=colors[i])
        bottom += fruit[i]

    plt.ylabel('Quantity of Fruit')
    plt.yticks(np.arange(0, 81, 10))
    plt.title('Number of Fruit per Person')
    plt.legend()
