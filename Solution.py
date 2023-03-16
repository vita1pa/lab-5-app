import warnings
import math
import pandas as pd
import numpy as np
from itertools import product, combinations


class Solution:
    def __init__(self, params):
        #кількість ітерацій та різні типи похибок - за замовчуванням
        self.num_iterations = 100000
        self.l1 = 0
        self.l2 = 0
        self.l3 = 0
        self.l4 = 0
        
        #кількість експертів
        self.m = params['m']
        assert self.m > 0, "Введіть додатне число експертів"

        #кількість альтернатив
        self.n = params['n']
        assert self.n > 0, "Введіть додатню кількість альтернатив"

        #коефіцієнти довіри до експертів
        self.weights = np.array(params['weights'])
        assert len(self.weights) == self.m, "Кількість коефіцієнтів довіри не рівна загальній кількості експертів"
        
        #оцінки експертів по кожній з альтернатив (матриця)
        self.estimations = np.array(params['estimations'])
        assert self.estimations.shape == (self.n, self.m), "Розмірність матриці оцінок - не збігається з числом експертів або/і числом альтернатив"  

    def __count_aprior_probabilities(self, with_test = False, event = None):
        #апріорні ймовірності рхуються як серенє зважене оцінок експертів:
        self.aprior_probabilities = self.estimations@self.weights.T
        self.aprior_probabilities =  self.aprior_probabilities/np.sum(self.weights)
        
        if with_test:
            self.aprior_probabilities[event] = 0.99999999
            
        # ймовірності протилежних подій
        self.aprior_probabilities_not = 1 - self.aprior_probabilities.copy()
        
        #ймовірності, яка застосоуватимуться в методі Монте-Карло (для запису середніх значень)
        self.probabilities = self.aprior_probabilities.copy()

    def __count_aprior_odds(self):
        #апріорні шанси - рахуються задопомогою вищенаведених ймовірностей
        self.aprior_odds = self.aprior_probabilities/(1 - self.aprior_probabilities)
        
        #апріорні шанси ненастання
        self.aprior_odds_not = 1/self.aprior_odds.copy()
        
        #шаснси, які застосовуватимуться в методі Монте-Карло
        self.odds = self.aprior_odds.copy()

    def __initialize_conditional_probabilities(self):
        #ініціалізуємо випадково, враховуючи обмеження з апріорних ймовірностей
        self.conditional_probabilities = np.ones((self.n, self.n))
        self.conditional_probabilities_not = np.ones((self.n, self.n))
        for i in range(self.n):
            for j in range(self.n):
                #Діагональні елементи не цікавлять, вважаємо одиницею (якщо відбулася подія a -то відбуласа подія a)
                if i == j:
                    continue

                #Визначаємо вищу та нижчу межі для генерації випадкового значення
                low = (self.aprior_probabilities[i]-1+self.aprior_probabilities[j])/self.aprior_probabilities[j]
                
                high = self.aprior_probabilities[i]/self.aprior_probabilities[j]

                if high>=1:
                    high = 1

                if low <=0:
                    low = 0
       
                self.conditional_probabilities[i,j] = np.random.uniform(low = low,high = high)
                
                denom = self.aprior_probabilities_not[j]
                nom = (self.aprior_probabilities[i] - self.aprior_probabilities[j]*self.conditional_probabilities[i,j])
                self.conditional_probabilities_not[i,j]= nom/denom
                

    def __initialize_conditional_odds(self):
        #блок, якщо настання
        temp_matrix = self.conditional_probabilities.copy()
        temp_matrix[temp_matrix==1]=0
        self.conditional_odds = temp_matrix/(1-temp_matrix)
        
        #блок ненастання
        temp_matrix = self.conditional_probabilities_not.copy()
        temp_matrix[temp_matrix==1]=0
        self.conditional_odds_not = temp_matrix/(1-temp_matrix)

    def initialize(self):
        #повна ініціалізація всього
        self.__count_aprior_probabilities()
        self.__count_aprior_odds()
        self.__initialize_conditional_probabilities()
        self.__initialize_conditional_odds()
        
        
    def __is_happened(self, probability):
        #Допоміжна функція для визначення, чи настане подія. 
        #Випадково обирається число в межах від 1 та 0, якщо менше ймовірності - то виконується
        threshold = np.random.uniform(low= 0, high = 1)
        return probability >= threshold
    
    def one_bunch_iteration(self, n):
        # Визначаємо, які з подій відбудуться на цьому кроці
        checklist = np.array([self.__is_happened(element) for element in self.aprior_probabilities ])
        
        #Оновлюємо шанси для інших подій при настанні (чи не настанні інших):
        for i in range(self.n):
            if checklist[i]:
                self.odds += self.aprior_odds*self.conditional_odds[:,i]
            else:
                self.odds += self.aprior_odds*self.conditional_odds_not[:, i]
            
        self.odds = self.odds/len(checklist)
        
        assert n>=1, "Введене значення n - менше за 1"
        self.probabilities = self.probabilities*(n-1)
        self.probabilities += self.odds.copy()/(self.odds.copy() + 1)
        self.probabilities = self.probabilities/n
        return checklist
    
    def process(self, num_iterations = 1000, vocab = True):
        self.num_iterations = num_iterations
        granularity = self.num_iterations/100
        for i in range(self.num_iterations):
            checklist = self.one_bunch_iteration(i+1)
            if vocab and i%granularity == 0:
                print("Probabilities for step №"+str(i)+" : " + str(self.probabilities)+ ";")
                
        self.new_aprior_probabilities = self.probabilities.copy()
    
    def __process_sigmas(self):
        self.sigmas = (self.m*np.sum(self.estimations**2, axis = 1) - np.sum(self.estimations, axis = 1)**2)
        self.sigmas = self.sigmas/(self.m*(self.m-1))
        self.sigmas = np.sqrt(self.sigmas)
        
    def __estimate_low_prob(self,expert):
        p_low = self.aprior_probabilities.copy()
        prob = self.estimations[:,expert]
        
        for i in range(self.n):
            p = [0, prob[i] - self.sigmas[i]]
            
            max_1 = 0
            for j in range(self.n):
                if i == j: 
                    continue
                    
                c_1 = prob[i]*(1 - self.conditional_probabilities[j, i])
                
                if c_1 > max_1:
                    max_1 = c_1
                
            p.append(max_1)
            
            p_low[i] = np.max(p)
        return p_low
    
    def __estimate_high_prob(self, expert):
        p_high = self.aprior_probabilities.copy()
        prob = self.estimations[:,expert]
        
        for i in range(self.n):
            p = [1, prob[i] + self.sigmas[i]]
            
            min_1 = 1
            min_2 = 1
            for j in range(self.n):
                if i == j: 
                    continue
                    
                c_1 = 1 - prob[i]*(1 - self.conditional_probabilities[j, i])
                
                if prob[i]  < self.conditional_probabilities[j, i]:
                    c_2 = prob[i]/self.conditional_probabilities[j, i]
                else:
                    c_2 = (1 - prob[i])/(1 - self.conditional_probabilities[j, i])
                    
                if c_1 < min_1:
                    min_1 = c_1
                    
                if c_2 < min_2:
                    min_2 = c_2
                
            p.append(min_1)
            p.append(min_2)
            
            p_high[i] = np.min(p)
            
        return p_high
        
    def estimate_truth_coeficient(self, vocab = True):
        self.__process_sigmas()
        
        high_probs = []
        for i in range(self.m):
            res = self.__estimate_high_prob(i)
            high_probs.append(res)
        
        high_probs = np.array(high_probs)
        
        low_probs = []
        for i in range(self.m):
            res = self.__estimate_low_prob(i)
            low_probs.append(res)
        
        low_probs = np.array(low_probs)
        
        high_odds = high_probs/(1-high_probs)
        low_odds = low_probs/(1-low_probs)
        
        to_max_1 = np.abs(1 - high_odds/self.aprior_odds)
        to_max_2 = np.abs(1 - low_odds/self.aprior_odds)
        
        #оцінка L1 - вплива похибки апріорної ймовірності
        self.l1 = 0
        for j in range(self.m):
            max_1 = 0
            max_2 = 0
            for i in range(self.n):
                
                if to_max_1[j, i] > max_1:
                    max_1 = to_max_1[j, i]
                
                if to_max_2[j, i] > max_2:
                    max_2 = to_max_2[j, i]
            self.l1 +=  (max_1+max_2)/(2*self.weights[j]*self.m) 
        
        #оцінки l2 -  похибка при малоймовірних подіях
        self.l2 = 0
        
        num_events = 100
        elementary_probability = 0.05
        prob_when_add = self.aprior_probabilities.copy()
        
        for i in range(num_events):
            to_max = 0
            conditional_probs_for_new = np.zeros(self.n)
            for j in range(self.n):
                low_1 = (self.aprior_probabilities[j] -1+elementary_probability)/elementary_probability
                low_2 = (self.aprior_odds[j]/3)/(self.aprior_odds[j]/3+1)
                high_1 = self.aprior_probabilities[j]/elementary_probability
                high_2 = (3*self.aprior_odds[j])/(3*self.aprior_odds[j]+1)
                
                low = np.max(np.array([0,low_1, low_2]))
                high = np.min(np.array([1,high_1, high_2]))
                
                conditional_probs_for_new[j] = np.random.uniform(low = low,high = high)
                
                prob_when_add[j] = prob_when_add[j]*(1-elementary_probability)+conditional_probs_for_new[j]*elementary_probability
                value = np.abs(1 - prob_when_add[j]/self.aprior_probabilities[j])
                if value > to_max:
                    to_max = value
            self.l2 += to_max
        
        self.l2 = self.l2/(num_events)
        
        #оцінки l3 - впливу маловпливової події
        elementary_probability = 0.5
        self.l3 = 0
        
        for i in range(num_events):
            to_max = 0
            conditional_probs_for_new = np.zeros(self.n)
            for j in range(self.n):
                low_1 = (self.aprior_probabilities[j] -1+elementary_probability)/elementary_probability
                low_2 = (self.aprior_odds[j]/1.3)/(self.aprior_odds[j]/1.3+1)
                high_1 = self.aprior_probabilities[j]/elementary_probability
                high_2 = (1.3*self.aprior_odds[j])/(1.3*self.aprior_odds[j]+1)
                
                low = np.max(np.array([0,low_1, low_2]))
                high = np.min(np.array([1,high_1, high_2]))
                
                conditional_probs_for_new[j] = np.random.uniform(low = low,high = high)
                
                prob_when_add[j] = prob_when_add[j]*(1-elementary_probability)+conditional_probs_for_new[j]*elementary_probability
                value = np.abs(1 - prob_when_add[j]/self.aprior_probabilities[j])
                if value > to_max:
                    to_max = value
            self.l3 += to_max
            
        self.l3 = self.l3/(num_events)
        
        #Похибка методу Монте-Карло
        self.l4 = 3/np.sqrt(self.num_iterations)
        
        #Коефіцієнт достовірності
        self.coef = (1 - self.l1)*(1 - self.l2)*(1-self.l3)*(1-self.l4)
        
        return self.l1*100, self.l2*100, self.l3*100, self.l4*100, self.coef*100
    
    def test_prob_check(self, event, vocab = True, num_iterations = 10000):    
        self.__count_aprior_probabilities(with_test = True, event = event)
        self.__count_aprior_odds()
        self.process(num_iterations = num_iterations, vocab=vocab)
        return self.probabilities
