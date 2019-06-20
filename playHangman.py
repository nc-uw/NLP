#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 8 13:13:25 2018

@author: nc57
"""
from __future__ import division
import random
import csv
import pickle
from collections import defaultdict
from string import ascii_lowercase
#import editdistance
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import math

print ('\n..reading csv file "words_250000_train.txt" as dataframe')
df = pd.read_csv("words_250000_train.txt", header=None)
print ('\n..df top5 obs..', df.head(5))
print ('\n..df bottom5 obs..', df.tail(5))
words = df.values

words_train, words_test = train_test_split(words, test_size=0.3, random_state=7)

df_train = pd.DataFrame(words_train)
df_train.to_csv('words_train.txt', index=None, header=None)

df_test = pd.DataFrame(words_test)
df_test.to_csv('words_test.txt', index=None, header=None)
df_train.head(50).to_csv('words_top50.txt', index=None, header=None)

#function that stores n-gram counts based on training data .. 
# .. where n is user-specified by parameter N
def train(ffile, N):
    
    count = 0
    f = open(ffile)
    
    #initialize a generalized n-gram dictionary
    model = {}
    for n in range(1,N+1):
        model[str(n)+'_gram'] = defaultdict(int)
    
    #loop over training data
    print('\n')    
    for line in f:
        
        count+=1
        #use progress bar
        if count%100000 == 0:
            print (count, ' words trained..')
            
        word = line.strip()
        #word = 'phonometric'
        #print (word)
        
        for n in range(1,N+1):
            try:
                for i in range(len(word)):
                    ngram_key = word[i:i+n]
                    if len(ngram_key) == n:
                        model[str(n)+'_gram'][ngram_key] +=1 
            except:
                continue
        '''    
        for n in range(1,N+1):
            total = sum(model[str(n)+'_gram'].values(), 0.0)
            model[str(n)+'_gram'] = {k: v / total for k, v in model[str(n)+'_gram'].items()}
        '''
    print (count, ' words trained..\n')
    return model


def initial_eval(guesses, model):
    
    #based on top5 unigram count in dictionary
    alphabets, count = zip(*list(model['1_gram'].items()))
    alphabets = np.array(alphabets)
    count = np.array(count)
    options = alphabets[count.argsort()][::-1]
    
    for o in options:
        if o not in guesses:
            return o
        

def ngram_eval(current, guesses, model, N):
    
    ngram_prob = {}    
    for n in range(2,N+1):
        prob = np.zeros(27)
        ##print ('n_', n, 'current', current)
        #loop over n-1 characters in 'current'
        for i in range(len(current)-(n-1)):            
            #extract ith ngram subset from 'current'
            subset = current[i:i+n]
            countN = np.zeros(27)
            #check if the subset is valid for prediction from ngram model
            #a subset is valid if only one character in subset is '_'
            if subset.count('_') == 1:
                ##print('valid subset', subset)
                #for all valid subsets compute counts
                for idx, alpha in enumerate(ascii_lowercase):
                    try_subset = subset.replace('_', alpha)
                    if model[str(n)+'_gram'][try_subset] > 0 and alpha not in guesses:    
                        countN[idx] = model[str(n)+'_gram'][try_subset]
                
            #compute prob for a given ngram
            if sum(countN) > 0:
                prob += countN/sum(countN)
            #print('prob', prob)
        
        #store prob in respective ngram key
        ngram_prob[str(n)+'_gram'] = prob
        
    return ngram_prob

def interpolation(ngram_prob):
    interp_prob = sum(ngram_prob.values())
    max_prob = max(interp_prob)
    if max_prob == 0: print ('max prob is 0!')
    pred = ascii_lowercase[np.where(interp_prob == max_prob)[0][0]]
    return pred


def play_hangman(model, word, N):  
    
    incorrect = 0
    guesses = []
    current = '_' * len(word)
    
    while incorrect < 6:
        
        if current.count('_') == len(word):
            pred = initial_eval(guesses, model)
        else:
            ngram_prob = ngram_eval(current, guesses, model, N)
            pred = interpolation(ngram_prob)
        #check if predicated character 'pred' is present in word
        if word.find(pred) == -1:
            #if not, increment incorrect count
            incorrect += 1
        else:
            #if it is, replace 'current' with 'pred' at all positions
            for i in range(0, len(word)):
                if word[i] == pred:
                    current_list = list(current)
                    current_list[i] = pred
                    current = ''.join(current_list)
        
        #append any guesses made till now
        guesses.append(pred)
        if current == word:
            break
    
    return current, guesses


#check accuracy on train and test set
def accuracy(model, ffile, N):
    f = open(ffile)
    correct = 0
    count = 0
    for line in f:
        word = line.rstrip()
        attempt, guesses = play_hangman(model, word, N) 
        #print (attempt, word, guesses)
        count += 1
        if attempt == word:
            correct += 1
            #print (correct)    
    print ('..accuracy % - ', float(correct/count))
    
# initialize model
N=6
model = train("words_train.txt", N)

# check accuracy
print ('\n..top50 performance stats')
accuracy(model, "words_top50.txt", N)
print ('\n..test performance stats')
accuracy(model, "words_test.txt", N)
print ('\n..train performance stats')
accuracy(model, "words_train.txt", N)
