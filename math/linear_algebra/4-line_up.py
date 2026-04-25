#!/usr/bin/python3
"""
Bu modul iki siyahının elementlərini toplamaq üçün funksiyanı ehtiva edir.
"""


def add_arrays(arr1, arr2):
    """
    İki siyahının elementlərini bir-birinə toplayır.
    Əgər siyahıların uzunluğu fərqlidirsə, None qaytarır.
    """
    if len(arr1) != len(arr2):
        return None
    return [arr1[i] + arr2[i] for i in range(len(arr1))]
