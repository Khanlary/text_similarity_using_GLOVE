# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 06:37:54 2020

@author: abhiom
"""

import pandas as pd
import numpy as np
import copy
import math
import os
import sys
import traceback
import logging
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', 
                    filename='log.txt', filemode='w', level=logging.DEBUG, 
                    datefmt='%Y-%m-%d %H:%M:%S')

PATH_GLOVE_VECTORS = "D:/Datasets/glove.6B/"
LIST_GLOVE_FILES = ["glove.6B.50d", "glove.6B.100d", "glove.6B.200d", "glove.6B.300d"]

def read_word_vectors_as_text(vector_file_index=1):
    """
    Read GLOVE vectors from the appropriate file using vector_file_index as the
    index over the list of file names stored in LIST_GLOVE_FILES.

    Parameters
    ----------
    vector_file_index : int, optional
        The index to select the name of the file to be read from the list of 
        files in LIST_GLOVE_FILES. The default is 1.

    Returns
    -------
    word_vectors_text : str
        The content of the file that is read or None is file doesn't exist'
    """
    #Combining filename and path to create the final path of the file to be read    
    word_vector_path = PATH_GLOVE_VECTORS + LIST_GLOVE_FILES[vector_file_index]
    logging.info("Path of word vector file is :"+ word_vector_path)
    logging.info("Reading :"+ word_vector_path)
    
    if os.path.exists(word_vector_path+".txt"):
        with open(word_vector_path+".txt", "r", encoding='utf-8') as f:
            try:
                word_vectors_text = f.read()
                if len(word_vectors_text):
                    logging.info("file read successfully!")
                else:
                    logging.warning("empty file read")
                return word_vectors_text    
            except Exception as e:
                logging.error(traceback.format_exc(e))
    else:
        logging.warning("The file doesn't exist")
        return None    

def create_word_vector_dictionary_from_text(word_vectors_text):
    """
    create word vector for each word by parsing the text file 

    Parameters
    ----------
    word_vectors_text : str
        Content of word to vector file in text format.

    Returns
    -------
    word_vector_dict : dict
        Dictionary with word as keys and value as its corrosponding vector.

    """
    word_vector_dict={}
    lines = word_vectors_text.split('\n')
    for line in lines:
        temp = line.split(" ")
        word_vector_dict[temp[0]] = temp[1:]
    return word_vector_dict

if __name__ == "__main__":
    word_vectors_text = read_word_vectors_as_text(vector_file_index=1)
    word_vector_dict = create_word_vector_dictionary_from_text(word_vectors_text)