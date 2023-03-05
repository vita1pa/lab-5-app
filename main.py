import streamlit as st
import numpy as np
import pandas as pd
import warnings
from Solution import *

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title='Task 5', 
    layout='wide'
)

st.title('The method of cross-influence analysis')

col1, col2 = st.columns(2)

probabilities_file = col1.file_uploader('Estimations input', type=['xlsx'], key='input_file')
weights_file = col1.file_uploader('Weights input', type=['xlsx'], key='weights_file')
num_iterations = col1.number_input('Number of iterations', value=100, step=1, key='number_of_iterations')

if col2.button('Run', key='run'):
    if (probabilities_file is None) or (weights_file is None):
        col2.error('Something is wrong in file. Review it and re-upload')
    else:
        res_cols = st.columns(7)
        col_names = [
            'Aprior estimated $$p_i$$:',
            'Aprior estimated $$p_i$$ with respect to relations',
            '$$L_1$$ error ( estimations in aprior probabilities error ) (%)',
            '$$L_2$$ error ( low probability events influence error ) (%)',
            '$$L_3$$ error ( independent events influence error ) (%)',
            '$$L_4$$ error ( Monte-Carlo scenario simulation error ) (%)',
            'Trust coefficient (%)'   
        ]
        estimations = pd.read_excel(probabilities_file)
        
        res_cols[0].write(weights)
        res_cols[0].write(estimations)
        
        weights = pd.read_excel(weights_file).T
        p = {
            'm':weights.shape[0],
            'n':estimations.shape[0],
            'weights': weights,
            'estimations': estimations
        }
        
        solution = Solution(p)
        solution.initialize()
        solution.process(num_iterations = num_iterations, vocab = False)
        l1, l2, l3, l4, P = solution.estimate_truth_coeficient(vocab = False)
        
        dataframes = [solution.aprior_probabilities, solution.new_aprior_probabilities, l1, l2, l3, l4, P]
        
        for i in range(7):
            res_cols[i].write(col_names[i])
            res_cols[i].dataframe(
                dataframes[i].style
                .format(precision=4)
                .applymap(lambda x: 'color: transparent' if pd.isnull(x) else '')
                .applymap(lambda x: 'background-color: transparent' if pd.isnull(x) else '')
            )
        


       