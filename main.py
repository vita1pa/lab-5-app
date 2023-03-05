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

probabilities_file = col1.file_uploader('Input', type=['xlsx'], key='input_file')
weights_file = col1.file_uploader('Weights', type=['xlsx'], key='weights_file')

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
        
        print(len(probabilities_file))
        print(len(weights_file))
        p = {'m':2, 
            'n':3, 
            'weights':[1, 1], 
            'estimations': [[0.3, 0.5],[ 0.7, 0.5], [ 0.4, 0.1]]
        }
        solution = Solution(p)
        
        


       
