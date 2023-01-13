# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 15:49:34 2022

@author: FRF8
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


def EDA(data):
    "This function plots standardised data using ECDF together with quantiles and CDF"
    
    standardised = data.filter(regex='z$')       
    # standardised = standardised.apply(lambda x: x.str.replace(',', '.'))
    standardised = standardised.apply(pd.to_numeric)
    
    # ex_ut = ""
    # ex_ut = ex_ut.filter(regex='z$')       
    # ex_ut = ex_ut.apply(lambda x: x.str.replace(',', '.'))
    # ex_ut = ex_ut.apply(pd.to_numeric)
    
    def ecdf(data):
        x = np.sort(data)
        y = np.arange(1, len(x)+1) / len(x)
        return(x, y)
    
    table_count = 1.1

    for columns in standardised:
        
        "Drop NA"
        a = standardised[columns].dropna()
        
        """
        Code to drop outliers +- 3 SD", not used
        
        """
        # greater_than = standardised[columns] > -3
        # less_than = standardised[columns] < 3
        # a = a[greater_than & less_than]
        
        # b = ex_ut[columns].dropna()
        # b_greater_than = ex_ut[columns] > -3
        # b_less_than = ex_ut[columns] < 3
        # b = b[b_greater_than & b_less_than]
                
        "Plot ECDF"
        x, y = ecdf(a)
        plt.plot(x, y, marker='.', linestyle='none', label='Empirical CDF')
        
        # ut_x, ut_y = ecdf(b)
        # plt.plot(ut_x, ut_y, marker='.', linestyle='none', label='Ex-ut')
        
        "Plot theoretical CDF"
        t_mean = np.mean(a)
        t_std = np.std(a)
        t_samples = np.random.normal(t_mean, t_std, size=10000)
        x_theor, y_theor = ecdf(t_samples)
        plt.plot(x_theor, y_theor, label='Theoretical CDF')
        
        "Plot quantiles"
        percentiles = np.array([25, 50, 75])
        percs = [0.25, 0.5, 0.75]
        perc_val = np.percentile(a, percentiles)
        plt.plot(perc_val, percs, marker='o', linestyle='none', color='red')
        
        z = 0
        percentiles = ['25th percentile', '50th percentile', '75th percentile']
        for x in percs:
            y = perc_val[z]
            percentile_count = percentiles[z]
            plt.text(y+0.3, x, percentile_count)
            z = z + 1
        
        "Display labels, set margins, ticks, show plot"
        plt.xlabel('Z scores')
        plt.ylabel('ECDF')
        plt.title("Figure "+str(table_count)+" Distribution of scores for "+columns, loc='left')

        plt.margins(0.02)
        plt.xticks([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
        plt.legend()
        table_count = round(table_count + 0.1, 1)
        plt.show()
