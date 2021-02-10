import time
import sys
import numpy as np
import pandas as pd
import logging
import pickle
import json


'''
SentimentUtilityLogisticModel class implements
    Sentiment Utility Logistic Model (SULM).
This model estimates users' and items' profiles based on information extracted
from user reviews.
The data should contain:
    - userID
    - itemID
    - overallRating - {0,1}
    - list of aspects sentiments - {0,1,nan}
'''
class SentimentUtilityLogisticModel():
    '''
    ratings - the list of data points 
    num_aspects - number of aspects in the dataset
    num_factors - number of latent factors in the model
    lambda_b, lambda_pq - regularization parameter for profile coefficients (default=0.6)
    lambda_z, lambda_w  - regularization parameter for regression weight (default=0.6)
    gamma - the coefficient for the initial gradient descent step (default=1.0)
    iterations - number of iterations for training the model (default=30)
    alpha - the relative importance between rating and sentiment estimation parts (default=0.5)
    l1 - L1 normalization
    l2 - L2 normalization
    mult - multiplication of general-user-item coefficients
    '''
    def __init__(self, logger, ratings, num_aspects,
                 num_factors=3,
                 lambda_b=0.5,
                 lambda_pq=0.5,
                 lambda_z=0.5,
                 lambda_w=0.5,
                 lambda_su=0.05,
                 gamma=1.0,
                 iterations=30,
                 alpha=0.5,
                 l1=False,
                 l2=True,
                 mult=False):

        self.logger = logger
        self.ratings = ratings
        self.num_ratings = len(ratings)
        self.num_aspects = num_aspects
        self.num_factors = num_factors
        self.iterations = iterations
        self.alpha = alpha
        self.lambda_b = lambda_b
        self.lambda_pq = lambda_pq
        self.lambda_z = lambda_z
        self.lambda_w = lambda_w
        self.lambda_su = lambda_su
        self.gamma = gamma
        self.mu = None
        self.l1 = l1
        self.l2 = l2
        self.mult = mult
        self.average_sentiments()

    '''
        Create new profile
        user:   True - user profile; False - item profile
        random: True - random initial coefficients; False: profile coefficiets set to zeros
    '''
    def new_profile(self, profile_id, user=True, random=True):
        if user:
            self.profile_users[profile_id] = dict()
            if random:
                self.profile_users[profile_id]['bu'] = np.random.normal(loc=0.0, scale=0.1,
                                                                        size=(self.num_aspects+1))
                self.profile_users[profile_id]['p']  = np.random.normal(loc=0.0, scale=0.1,
                                                                        size=(self.num_aspects+1,self.num_factors))
                self.profile_users[profile_id]['w']  = np.random.normal(loc=0.0, scale=0.1,
                                                                        size=(self.num_aspects+1))
            else:
                self.profile_users[profile_id]['bu'] = np.zeros(shape=(self.num_aspects+1))
                self.profile_users[profile_id]['p']  = np.zeros(shape=(self.num_aspects+1,self.num_factors))
                self.profile_users[profile_id]['w']  = np.zeros(shape=(self.num_aspects+1))
        else:
            self.profile_items[profile_id] = dict()
            if random:
                self.profile_items[profile_id]['bi'] = np.random.normal(loc=0.0, scale=0.1,
                                                                        size=(self.num_aspects+1))
                self.profile_items[profile_id]['q']  = np.random.normal(loc=0.0, scale=0.1,
                                                                        size=(self.num_aspects+1,self.num_factors))
                self.profile_items[profile_id]['v']  = np.random.normal(loc=0.0, scale=0.1,
                                                                        size=(self.num_aspects+1))
            else:
                self.profile_items[profile_id]['bi'] = np.zeros(shape=(self.num_aspects+1))
                self.profile_items[profile_id]['q']  = np.zeros(shape=(self.num_aspects+1,self.num_factors))
                self.profile_items[profile_id]['v']  = np.zeros(shape=(self.num_aspects+1))
    
    def new_variable_profile(self, profile_id, user=True, random=True):
        variable_profile = dict()
        if user:
            variable_profile['bu'] = np.zeros(shape=(self.num_aspects+1))
            variable_profile['p']  = np.zeros(shape=(self.num_aspects+1,self.num_factors))
            variable_profile['w']  = np.zeros(shape=(self.num_aspects+1))
        else:
            variable_profile['bi'] = np.zeros(shape=(self.num_aspects+1))
            variable_profile['q']  = np.zeros(shape=(self.num_aspects+1,self.num_factors))
            variable_profile['v']  = np.zeros(shape=(self.num_aspects+1))
        return variable_profile.copy()
    
    
    
    '''Calculate average \mu for parameter initialization'''
    def mu_initialization(self):
        if self.mu:
            return
        
        'iterate over aspects'
        aspects = zip(*self.ratings)
        self.mu = list()
        for i, aspect in enumerate(aspects):
            'rating'
            if i ==2:
                mean_rating = self.logistic_inverse(np.nanmean(aspect))
            'dismiss user_id, item_id, rating'
            if i < 3:
                continue
            self.mu.append(self.logistic_inverse(np.nanmean(aspect)))
        'last mu is for constant'
        self.mu.append(mean_rating)
        self.mu = np.array(self.mu)

    '''Calculate average sentiments'''
    def average_sentiments(self):
        aspects = zip(*self.ratings)
        self.avg_sentiments = list()
        for i, aspect in enumerate(aspects):
            'dismiss user_id, item_id, rating'
            if i < 3:
                continue
            self.avg_sentiments.append(np.nanmean(aspect))

    '''Calculate correlation between sentiments'''

    '''Train the model to fit the rating data set'''
    def train_model(self, l1 = False, l2 = True):
        #initialize coefficients
        self.mu_initialization() #initialize with average values
#         self.mu = np.random.normal(size=(self.num_aspects+1)) #random initialization
        self.logger.info('Initial mu: %s'%str(self.mu))
        self.z  = np.random.normal(loc=(1.0/self.num_aspects), scale=0.1, size=(self.num_aspects+1)) #random initialization
        self.profile_users = dict()
        self.profile_items = dict()
        Q_old = 100000000000000000000.0
        conv_num = 0
        #make the specified number of iterations
        for i in range(self.iterations):
            t0 = time.time()
            #self.ratings - the list of arrays
            #shuffle the list of ratings on each iteration
            np.random.shuffle(self.ratings)
            for num, element in enumerate(self.ratings):
                user = element[0]
                item = element[1]
                if user not in self.profile_users:
                    self.new_profile(user, user=True)
                if item not in self.profile_items:
                    self.new_profile(item, user=False)
                    
                rating = element[2]
                aspect_ratings = np.append(element[3:],np.nan)
                assert len(aspect_ratings) == self.num_aspects + 1

                # identify which aspects are specified
                indicator = np.invert(np.isnan(aspect_ratings))

                #calculate aspect sentiment predictions
                sentiment_utility_prediction  = self.calculate_sentiment_utility_prediction(user, item)
                # sentiment_utility_prediction_initial = sentiment_utility_prediction
                sentiment_prediction  = self.logistic(sentiment_utility_prediction)
                #calculate rating predictions
                if self.mult:
                    rating_utility_prediction = sum(sentiment_utility_prediction * (self.z * self.profile_users[user]['w'] * self.profile_items[item]['v']))
                else:
                    rating_utility_prediction = sum(sentiment_utility_prediction * (self.z + self.profile_users[user]['w'] + self.profile_items[item]['v']))
                rating_prediction = self.logistic(rating_utility_prediction)
                
                # calculate deltas
                delta_s = aspect_ratings - sentiment_prediction
                delta_s = delta_s - (np.abs(delta_s) < 0.001)*delta_s
                delta_s = delta_s - 0.001*(np.abs(delta_s) > 0.999)*delta_s
                
                delta_r = rating - rating_prediction
                delta_r = delta_r - (np.abs(delta_r) < 0.001)*delta_r
                delta_r = delta_r - 0.001*(np.abs(delta_r) > 0.999)*delta_r
                
                # update vector mu
                if self.mult:
                    mu_step = self.alpha * delta_r * (self.z * self.profile_users[user]['w'] * self.profile_items[item]['v'])
                else:
                    mu_step = self.alpha * delta_r * (self.z + self.profile_users[user]['w'] + self.profile_items[item]['v'])
                mu_step += (1 - self.alpha) * np.nan_to_num(indicator.astype(int) * delta_s)
                if any(np.abs(mu_step) > 1000):
                    print(mu_step)
                    print('mu_step',delta_r, self.z, self.profile_users[user]['w'], self.profile_items[item]['v'])
                    print()
                    exit()
                
                # Fix items and update users' profiles
                # take step towards negative gradient of log likelihood
                # we take a step in negative direction because we are minimizing functional Q
                bu_step = mu_step
                if self.l2:
                    self.profile_users[user]['bu'] -= self.gamma * self.lambda_b * self.profile_users[user]['bu']
                if self.l1:
                    self.profile_users[user]['bu'] -= self.gamma * self.lambda_b * np.sign(self.profile_users[user]['bu'])
                if self.lambda_su:
                    self.profile_users[user]['bu'] -= self.gamma * self.lambda_su * indicator.astype(int) * self.profile_users[user]['bu']
                
                self.profile_users[user]['bu'] += self.gamma * bu_step
                 
                p_step = np.matrix([np.dot(self.profile_items[item]['q'][i], mu_step[i]) for i in range(self.num_aspects+1)])
                if self.l2:
                    self.profile_users[user]['p'] -= self.gamma * self.lambda_pq * self.profile_users[user]['p']
                if self.l1:
                    self.profile_users[user]['p'] -= self.gamma * np.sign(self.profile_users[user]['p'])
                if self.lambda_su:
                    self.profile_users[user]['p'] -= self.gamma * self.lambda_su * np.matrix([np.dot(self.profile_items[item]['q'][i], indicator.astype(int)[i]) for i in range(self.num_aspects+1)]) 
                    
                
                self.profile_users[user]['p'] += self.gamma * p_step


                # Fix users and update items' profiles
                # take step towards negative gradient of log likelihood
                # we take a step in negative direction because we are minimizing functional Q
                #calculate aspect sentiment predictions
                sentiment_utility_prediction = self.calculate_sentiment_utility_prediction(user, item)
                sentiment_prediction = self.logistic(sentiment_utility_prediction)
                if self.mult:
                    rating_utility_prediction = sum(sentiment_utility_prediction * (self.z * self.profile_users[user]['w'] * self.profile_items[item]['v']))
                else:
                    rating_utility_prediction = sum(sentiment_utility_prediction * (self.z + self.profile_users[user]['w'] + self.profile_items[item]['v']))
                rating_prediction = self.logistic(rating_utility_prediction)
                 
                delta_s = aspect_ratings - sentiment_prediction
                delta_s = delta_s - (np.abs(delta_s) < 0.001)*delta_s
                delta_s = delta_s - 0.001*(np.abs(delta_s) > 0.999)*delta_s

                delta_r = rating - rating_prediction
                delta_r = delta_r - (np.abs(delta_r) < 0.001)*delta_r
                delta_r = delta_r - 0.001*(np.abs(delta_r) > 0.999)*delta_r
                
                # calculate mu_step
                if self.mult:
                    mu_step = self.alpha * delta_r * (self.z * self.profile_users[user]['w'] * self.profile_items[item]['v'])
                else:
                    mu_step = self.alpha * delta_r * (self.z + self.profile_users[user]['w'] + self.profile_items[item]['v'])
                mu_step += (1 - self.alpha) * np.nan_to_num(indicator.astype(int) * delta_s)
                
                bi_step = mu_step
                if self.l2:
                    self.profile_items[item]['bi'] -= self.gamma * self.lambda_b * self.profile_items[item]['bi']
                if self.l1:
                    self.profile_items[item]['bi'] -= self.gamma * self.lambda_b * np.sign(self.profile_items[item]['bi'])
                if self.lambda_su:
                    self.profile_items[item]['bi'] -= self.gamma * self.lambda_su * indicator.astype(int) * self.profile_items[item]['bi']
                
                self.profile_items[item]['bi'] += self.gamma * bi_step
                 
                q_step = np.matrix([np.dot(self.profile_users[user]['p'][i], mu_step[i]) for i in range(self.num_aspects+1)])
                if self.l2:
                    self.profile_items[item]['q'] -= self.gamma * self.lambda_pq * self.profile_items[item]['q']
                if self.l1:
                    self.profile_items[item]['q'] -= self.gamma * self.lambda_pq * np.sign(self.profile_items[item]['q'])
                if self.lambda_su:
                    self.profile_items[item]['q'] -= self.gamma * self.lambda_su * np.matrix([np.dot(self.profile_users[user]['p'][i], indicator.astype(int)[i]) for i in range(self.num_aspects+1)])
                    
                self.profile_items[item]['q'] += self.gamma * q_step
                
            
                # Fix users, items profiles and solve for weights
                # take step towards negative gradient of log likelihood
                # we take a step in negative direction because we are minimizing functional Q
                #calculate aspect sentiment predictions
                sentiment_utility_prediction  = self.calculate_sentiment_utility_prediction(user, item)
                sentiment_prediction  = self.logistic(sentiment_utility_prediction)
                #calculate rating predictions
                if self.mult:
                    rating_utility_prediction = sum(sentiment_utility_prediction*(self.z * self.profile_users[user]['w'] * self.profile_items[item]['v']))
                else:
                    rating_utility_prediction = sum(sentiment_utility_prediction*(self.z + self.profile_users[user]['w'] + self.profile_items[item]['v']))
                rating_prediction = self.logistic(rating_utility_prediction)
                
                
                delta_r = rating - rating_prediction
                delta_r = delta_r - (np.abs(delta_r) < 0.001)*delta_r
                delta_r = delta_r - 0.001*(np.abs(delta_r) > 0.999)*delta_r
                
                z_step = self.alpha * sentiment_utility_prediction * delta_r
                if self.mult:
                    z_step *= self.profile_users[user]['w']*self.profile_items[item]['v']
                
                if self.l2:
                    z_step -= self.lambda_z * self.z
                if self.l1:
                    z_step -= self.lambda_z * np.sign(self.z)
                
                w_step = self.alpha * sentiment_utility_prediction * delta_r
                if self.mult:
                    w_step *=  self.z*self.profile_items[item]['v']
                
                if self.l2:
                    w_step -= self.lambda_w * self.profile_users[user]['w']
                if self.l1:
                    w_step -= self.lambda_w * np.sign(self.profile_users[user]['w'])
                
                v_step = self.alpha * sentiment_utility_prediction * delta_r
                if self.mult:
                    v_step *= self.z*self.profile_users[user]['w']
                    
                if self.l2:
                    v_step -= self.lambda_w * self.profile_items[item]['v']
                if self.l1:
                    v_step -= self.lambda_w * np.sign(self.profile_items[item]['v'])
                
                
                self.z += self.gamma * z_step
                
                self.profile_users[user]['w'] += self.gamma * w_step
                
                self.profile_items[item]['v'] += self.gamma * v_step
                                 
                
                
                if num%10000==0 and num > 0:
                    self.logger.debug('%d elements processed'%num)

            #update the length of gradient descent step
            self.gamma *= 0.91
            t1 = time.time()
            Q_new = self.calculate_Q()
            Q_dif = (Q_old-Q_new)/Q_old
            Q_old = Q_new
            self.logger.info('Iteration %.2i finished in %.2f seconds with Q = %.3f (diff = %.4f)'% (i + 1, t1 - t0, Q_old, Q_dif))
            if Q_dif < 0.005 and Q_dif > 0:
                conv_num += 1
                if conv_num > 2:
                    self.logger.info('Model converged on iteration %.2i'%(i+1))
                    break
            else:
                conv_num = 0
    
    
    
    '''Calculate the aspect sentiments predictions based on user and item profile'''   
    def calculate_sentiment_utility_prediction(self, user, item):
        sentiment_utility_predictions  = self.mu.copy()
        sentiment_utility_predictions += self.profile_users[user]['bu']
        sentiment_utility_predictions += self.profile_items[item]['bi']
        product = [np.dot(self.profile_users[user]['p'][i],self.profile_items[item]['q'][i]) for i in range(self.num_aspects+1)]
        sentiment_utility_predictions += product
        return sentiment_utility_predictions
    
    
    '''Calculate the logistic function and its inverse'''
    def logistic(self,t):
        return 1/(1+np.exp(-t))
    def logistic_inverse(self,t):
        if t<0.00001:
            return -40
        return -np.log(1/t - 1)
    
    
    '''Calculate the value of the functional Q to be optimized'''
    def calculate_Q(self):
        rating_part = - self.log_likelihood_rating()
        sentiment_part = - self.log_likelihood_sentiment()
        rerularization_part = self.regularization()

        Q =  self.alpha * rating_part + (1 - self.alpha) * sentiment_part + rerularization_part
        if np.isnan(Q):
            print(rating_part, self.alpha, sentiment_part, rerularization_part)
        return Q
    
    
    '''Calculate log-likelihood for the sentiment part of the model'''
    def log_likelihood_sentiment(self):
        log_likelihood = 0
        for element in self.ratings:
            user = element[0]
            item = element[1]
            aspect_ratings = element[3:]
            
            indicator = np.invert(np.isnan(aspect_ratings))
            'calculate aspect sentiment predictions'
            s_logistic_predictions  = self.logistic(self.calculate_sentiment_utility_prediction(user, item))
#             print(s_logistic_predictions,aspect_ratings)
            for i in range(len(indicator)):
                if indicator[i]:
                    log_likelihood += aspect_ratings[i] * np.log(s_logistic_predictions[i])
                    log_likelihood += (1 - aspect_ratings[i]) * np.log(1 - s_logistic_predictions[i])
#             print('log_likelihood_sentiment',log_likelihood)
        return log_likelihood
    
    
    '''Calculate log-likelihood for the rating part of the model'''
    def log_likelihood_rating(self):
        log_likelihood = 0
        for element in self.ratings:
            user = element[0]
            item = element[1]
            rating = element[2]
            'calculate aspect sentiment predictions'
            s_predictions  = self.calculate_sentiment_utility_prediction(user, item)
            if self.mult:
                r_prediction = sum(s_predictions * (self.z * self.profile_users[user]['w'] * self.profile_items[item]['v']))
            else:
                r_prediction = sum(s_predictions * (self.z + self.profile_users[user]['w'] + self.profile_items[item]['v']))
            if np.isnan(r_prediction):
                print(s_predictions,self.z,self.profile_users[user]['w'],self.profile_items[item]['v'])
                exit()
            r_logistic_prediction = self.logistic(r_prediction)
            if r_logistic_prediction > 0.999:
                r_logistic_prediction = 0.999
            elif r_logistic_prediction < 0.001:
                r_logistic_prediction = 0.001
            log_likelihood += rating * np.log(r_logistic_prediction)
            log_likelihood += (1 - rating) * np.log(1-r_logistic_prediction)
            if np.isnan(log_likelihood):
                print(r_prediction,r_logistic_prediction)
                print(rating,np.log(r_logistic_prediction),np.log(1-r_logistic_prediction))
                exit()
                break
        return log_likelihood
    
    ''' Calculate the L2 regularization part of the model'''
    def regularization(self):
        user_norm = dict()
        item_norm = dict()
        
        if self.l2:
            norm_function = np.square
        elif self.l1:
            norm_function = np.abs
        
        norm_z = np.sum(norm_function(self.z))
        
        for user in self.profile_users:
            norm_b  = np.sum(norm_function(self.profile_users[user]['bu']))
            norm_pq = np.sum(norm_function(self.profile_users[user]['p']))
            norm_w  = np.sum(norm_function(self.profile_users[user]['w']))
            user_norm[user] = self.lambda_b * norm_b + self.lambda_pq * norm_pq +  self.lambda_w * norm_w
            
        for item in self.profile_items:
            norm_b  = np.sum(norm_function(self.profile_items[item]['bi']))
            norm_pq = np.sum(norm_function(self.profile_items[item]['q']))
            norm_w  = np.sum(norm_function(self.profile_items[item]['v']))
            item_norm[item] = self.lambda_b * norm_b + self.lambda_pq * norm_pq +  self.lambda_w * norm_w
        
        total_norm = 0
        for element in self.ratings:
            user = element[0]
            item = element[1]
            total_norm += user_norm[user] + item_norm[item] + self.lambda_z * norm_z
            
            aspect_ratings = np.append(element[3:],np.nan)
            indicator = np.invert(np.isnan(aspect_ratings))
            sentiment_utility_predictions  = indicator.astype(int) * self.calculate_sentiment_utility_prediction(user, item)
            total_norm += self.lambda_su * np.sum(np.square(sentiment_utility_predictions))
        return total_norm
    
    # Print train output ONLY for testing purposes 
    def predict_train(self):
        for i, element in enumerate(self.ratings):
            user = element[0]
            item = element[1]
#             rating = element[2]
            #calculate aspect sentiment predictions
            s_predictions  = self.calculate_sentiment_utility_prediction(user, item)
            s_logistic_predictions  = self.logistic(s_predictions)
            if self.mult:
                r_prediction = sum(s_predictions*(self.z * self.profile_users[user]['w'] * self.profile_items[item]['v']))
            else:
                r_prediction = sum(s_predictions*(self.z + self.profile_users[user]['w'] + self.profile_items[item]['v']))
            r_logistic_prediction = self.logistic(r_prediction)
            message = str(element) + '\nRating: %d\tPrediction: %.3f'%(element[2],r_logistic_prediction)
            message += '\nReal sentiments: %s\nPredicted sentiments: %s'%(element[3:],s_logistic_predictions[:-1])
            self.logger.info(message)
            if i > 15:
                break
            
    
    
    def predict(self, user, item):
        '''
        Predict ratings and sentiments for a pair of user and item
        Input:  user_id, item_id
        Output: rating_prediction, list of sentiment_predictions
        '''
        if user not in self.profile_users:
            self.new_profile(user, user=True, random=False)
        if item not in self.profile_items:
            self.new_profile(item, user=False, random=False)
            
        'calculate aspect sentiment predictions'
        sentiment_utility_prediction = self.calculate_sentiment_utility_prediction(user, item)
        sentiment_predictions  = self.logistic(sentiment_utility_prediction)
        if self.mult:
            rating_utility_prediction = sum(sentiment_utility_prediction * (self.z * self.profile_users[user]['w'] * self.profile_items[item]['v']))
        else:
            rating_utility_prediction = sum(sentiment_utility_prediction * (self.z + self.profile_users[user]['w'] + self.profile_items[item]['v']))
        rating_prediction = self.logistic(rating_utility_prediction)
        return rating_prediction, sentiment_predictions
    
    
    def calculate_aspect_impacts(self, user, item, average = False, absolute = True):
        '''Calculate aspect impacts for a given pair of user_id, item_id'''
        if user not in self.profile_users:
            self.new_profile(user, user = True, random = False)
        if item not in self.profile_items:
            self.new_profile(item, user = False, random = False)
            
        'calculate aspect sentiment predictions'
        sentiment_prediction  = self.logistic(self.calculate_sentiment_utility_prediction(user, item)[:-1])
        
        if self.mult:
            aspect_impacts = (self.z * self.profile_users[user]['w'] * self.profile_items[item]['v'])[:-1]    
        else:
            aspect_impacts = (self.z + self.profile_users[user]['w'] + self.profile_items[item]['v'])[:-1]

        if average:
            sentiment_difference = sentiment_prediction - self.avg_sentiments
            aspect_impacts = aspect_impacts * sentiment_difference
        else:
            aspect_impacts = aspect_impacts * sentiment_prediction
        
        if absolute:
            aspect_impacts =  np.abs(aspect_impacts)
        return list(aspect_impacts)
        
        
    
    '''Predict ratings and sentiments for a given dataset'''
    def predict_test(self, testset, filename):
        result = list()
        dictionary={}
        for element in testset:
            user = element[0]
            item = element[1]
            if user not in self.profile_users:
                self.new_profile(user, user=True, random=False)
            if item not in self.profile_items:
                self.new_profile(item, user=False, random=False)
            
            r_logistic_prediction, s_logistic_predictions = self.predict(user, item)
            result.append([user,item,r_logistic_prediction]+s_logistic_predictions.tolist())
            
            temp1=self.calculate_aspect_impacts(user, item, average = True, absolute = False)
            temp2=self.calculate_aspect_impacts(user, item, average = True, absolute = True)
            if not user in item:
                dictionary[user]={}
            dictionary[user][item]=temp1
            
            
            self.logger.info('User: %s, Item %s'%(user,item))
            self.logger.info('Aspect_impacts: '+
                             str(temp1))
            self.logger.info('ABSOLUTE_Aspect_impacts: '+
                             str(temp2))
            
        json.dump(result,open(filename,'w'))
        print(str(temp1.index(max(temp1)))+' is the best aspect for given user and item')
        #print(result,'**',dictionary,max_aspect)
        
        return result
    
    
    '''Calculate aspect index'''
    def calculate_aspect_index(self,aspect):
        if (aspect)=='FOOD':
            return 0
        if (aspect)=='SERVICE':
            return 1
        if (aspect)=='DECOR':
            return 2   
        
    '''Predicting best resturant for given aspect'''    
    def predict_best_rest_for_given_aspect(self,aspect):
        index=aspect
        max_item=''
        maxi=-99999
        for item in self.profile_items:
            avg_sum=0
            length=0
            for user in self.profile_users:
                avg_sum+=self.calculate_aspect_impacts(user,item,average=True,absolute=False)[index]
                length+=1
                avg_sum=avg_sum/length
                #print(avg_sum)
            if  avg_sum>maxi:
                maxi=avg_sum
                max_item=item
        print(max_item+' is the best restaurant for given aspect.')
        return max_item
                                
                
                
    '''Print the model to file in the readable format'''
    def pretty_save(self,filename):
        model_file = open(filename,'w')
        model_file.write('\mu = '+np.array_str(self.mu)+'\n')
        for user in self.profile_users:
            model_file.write('\n*****\n')
            model_file.write(user+'\nbu = '+np.array_str(self.profile_users[user]['bu'])+'\n')
            model_file.write('p = '+np.array_str(self.profile_users[user]['p'])+'\n')
            model_file.write('w = '+np.array_str(self.profile_users[user]['w'])+'\n')
        model_file.write('\n===========================\n===========================\n\n')
        for item in self.profile_items:
            model_file.write('\n*****\n')
            model_file.write(item+'\nbi = '+np.array_str(self.profile_items[item]['bi'])+'\n')
            model_file.write('q = '+np.array_str(self.profile_items[item]['q'])+'\n')
            model_file.write('v = '+np.array_str(self.profile_items[item]['v'])+'\n')
        model_file.write('\n===========================\n===========================\n\n')
        model_file.write('z = '+np.array_str(self.z)+'\n')
        model_file.close()
        
    '''Save the model'''
    def save(self, filename):
        pickle.dump(self.mu, open(filename+'mu', 'wb'))
        pickle.dump(self.avg_sentiments, open(filename+'av_sent', 'wb'))
        pickle.dump(self.z, open(filename+'z', 'wb'))
        pickle.dump(self.profile_users, open(filename+'user_profiles', 'wb'))
        pickle.dump(self.profile_items, open(filename+'item_profiles', 'wb'))
    
    '''Load the model'''
    def load(self, filename):
        self.mu = pickle.load(open(filename+'mu', 'rb'))
        self.avg_sentiments = pickle.load(open(filename+'av_sent', 'rb'))
        self.z = pickle.load(open(filename+'z', 'rb'))
        self.profile_users = pickle.load(open(filename+'user_profiles', 'rb'))
        self.profile_items = pickle.load(open(filename+'item_profiles', 'rb'))



     
if __name__ == '__main__':
    logger = logging.getLogger('signature')
    logging.basicConfig(format='%(asctime)s : %(name)-12s: %(levelname)s : %(message)s')
    logging.root.setLevel(level=logging.DEBUG)
    logger.info("running %s" % ' '.join(sys.argv))
    
    '''ratings = [['user1','item1',1,1,1,0],
               ['user1','item2',1,0,1,0],
               ['user2','item1',0,1,0,1],
               ['user2','item2',0,np.nan,0,1],
               ['user2','item3',1,1,0,1],
               ['user3','item1',1,np.nan,0,0],
               ['user3','item2',0,np.nan,0,1],
               ['user3','item3',1,1,1,1],
               ['user4','item3',1,0,1,1],
               ['user4','item1',0,1,0,1],
               ['user4','item2',1,0,1,1]
               ]'''
    
    
    
    arr2=["DRINKS", "DRINKS_ALCOHOL", "DRINKS_ALCOHOL_BEER", "DRINKS_ALCOHOL_HARD", "DRINKS_ALCOHOL_LIGHT", "DRINKS_ALCOHOL_WINE", "DRINKS_NON-ALCOHOL_COLD", "DRINKS_NON-ALCOHOL_HOT", "EXPERIENCE", "EXPERIENCE_BONUS", "EXPERIENCE_COMPANY", "EXPERIENCE_OCCASION", "EXPERIENCE_RECOMMENDATIONS", "EXPERIENCE_RESERVATION", "EXPERIENCE_TAKEOUT", "EXPERIENCE_TIME", "FOOD_FOOD", "FOOD_FOOD_BREAD", "FOOD_FOOD_CHEESE", "FOOD_FOOD_CHICKEN", "FOOD_FOOD_DESSERT", "FOOD_FOOD_DISH", "FOOD_FOOD_EGGS", "FOOD_FOOD_FRUIT", "FOOD_FOOD_MEAT", "FOOD_FOOD_MEAT_BACON", "FOOD_FOOD_MEAT_BEEF", "FOOD_FOOD_MEAT_BURGER", "FOOD_FOOD_MEAT_LAMB", "FOOD_FOOD_MEAT_PORK", "FOOD_FOOD_MEAT_RIB", "FOOD_FOOD_MEAT_STEAK", "FOOD_FOOD_MEAT_VEAL", "FOOD_FOOD_SALAD", "FOOD_FOOD_SAUCE", "FOOD_FOOD_SEAFOOD", "FOOD_FOOD_SEAFOOD_FISH", "FOOD_FOOD_SEAFOOD_SEA", "FOOD_FOOD_SIDE", "FOOD_FOOD_SIDE_PASTA", "FOOD_FOOD_SIDE_POTATO", "FOOD_FOOD_SIDE_RICE", "FOOD_FOOD_SIDE_VEGETABLES", "FOOD_FOOD_SOUP", "FOOD_FOOD_SUSHI", "FOOD_MEALTYPE_BREAKFAST", "FOOD_MEALTYPE_BRUNCH", "FOOD_MEALTYPE_DINNER", "FOOD_MEALTYPE_LUNCH", "FOOD_MEALTYPE_MAIN", "FOOD_MEALTYPE_START", "FOOD_PORTION", "FOOD_SELECTION", "GENERAL", "PERSONAL", "RESTAURANT", "RESTAURANT_ATMOSPHERE", "RESTAURANT_CUSINE", "RESTAURANT_ENTERTAINMENT_MUSIC", "RESTAURANT_ENTERTAINMENT_SPORT", "RESTAURANT_INTERIOR", "RESTAURANT_INTERNET", "RESTAURANT_LOCATION", "RESTAURANT_MONEY", "RESTAURANT_PARKING", "SERVICE"]
    import json
    import numpy as np
    from pprint import pprint

    with open('rest10000.json') as f:
        data1 = json.load(f)

    '''print(((data1[0]['aspects'])))'''

    with open('review10000.json') as f:
        data2 = json.load(f)

    '''pprint(((data2[0])))'''


    ratings=[]
    
    
    for j in range(len(data1)):
        temparr=['','','']
        for i1 in range(len(arr2)):
            temparr.append(np.nan)
        for x in range(len(data2)):
            if (data2[x]['review_id']==data1[j]['review_id']):
                tempdict=data1[j]['aspects']
                for valuess in tempdict.values():
                    for k,v in valuess.items():
                        for n,i in enumerate(arr2):
                            if (k==i and int(v)>=0):
                                temparr[3+n]=int(v)
                                temparr[0]=data2[x]['user_id']
                                temparr[1]=data2[x]['business_id']
                                if (data2[x]['stars']>3):
                                    temparr[2]=1
                                else:
                                    temparr[2]=0
        if temparr[0]!='':
            ratings.append(temparr)
        if j>2000:
            break
        #print(len(ratings[0]))
        
    print(len(ratings))
    #print(ratings[:100])
    
    def get_reviews(item_id):
        for data in  (data2):
            if item_id==data['business_id']:
                print(data['text'])
        for data in  (data1):
            if item_id==data['business_id']:
                print(data['aspects'])
    
    
    #ratings=[['a', 'b', 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], ['a1', 'b1', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    np.random.seed(241)
    aspects=["DRINKS", "DRINKS_ALCOHOL", "DRINKS_ALCOHOL_BEER", "DRINKS_ALCOHOL_HARD", "DRINKS_ALCOHOL_LIGHT", "DRINKS_ALCOHOL_WINE", "DRINKS_NON-ALCOHOL_COLD", "DRINKS_NON-ALCOHOL_HOT", "EXPERIENCE", "EXPERIENCE_BONUS", "EXPERIENCE_COMPANY", "EXPERIENCE_OCCASION", "EXPERIENCE_RECOMMENDATIONS", "EXPERIENCE_RESERVATION", "EXPERIENCE_TAKEOUT", "EXPERIENCE_TIME", "FOOD_FOOD", "FOOD_FOOD_BREAD", "FOOD_FOOD_CHEESE", "FOOD_FOOD_CHICKEN", "FOOD_FOOD_DESSERT", "FOOD_FOOD_DISH", "FOOD_FOOD_EGGS", "FOOD_FOOD_FRUIT", "FOOD_FOOD_MEAT", "FOOD_FOOD_MEAT_BACON", "FOOD_FOOD_MEAT_BEEF", "FOOD_FOOD_MEAT_BURGER", "FOOD_FOOD_MEAT_LAMB", "FOOD_FOOD_MEAT_PORK", "FOOD_FOOD_MEAT_RIB", "FOOD_FOOD_MEAT_STEAK", "FOOD_FOOD_MEAT_VEAL", "FOOD_FOOD_SALAD", "FOOD_FOOD_SAUCE", "FOOD_FOOD_SEAFOOD", "FOOD_FOOD_SEAFOOD_FISH", "FOOD_FOOD_SEAFOOD_SEA", "FOOD_FOOD_SIDE", "FOOD_FOOD_SIDE_PASTA", "FOOD_FOOD_SIDE_POTATO", "FOOD_FOOD_SIDE_RICE", "FOOD_FOOD_SIDE_VEGETABLES", "FOOD_FOOD_SOUP", "FOOD_FOOD_SUSHI", "FOOD_MEALTYPE_BREAKFAST", "FOOD_MEALTYPE_BRUNCH", "FOOD_MEALTYPE_DINNER", "FOOD_MEALTYPE_LUNCH", "FOOD_MEALTYPE_MAIN", "FOOD_MEALTYPE_START", "FOOD_PORTION", "FOOD_SELECTION", "GENERAL", "PERSONAL", "RESTAURANT", "RESTAURANT_ATMOSPHERE", "RESTAURANT_CUSINE", "RESTAURANT_ENTERTAINMENT_MUSIC", "RESTAURANT_ENTERTAINMENT_SPORT", "RESTAURANT_INTERIOR", "RESTAURANT_INTERNET", "RESTAURANT_LOCATION", "RESTAURANT_MONEY", "RESTAURANT_PARKING", "SERVICE"]
    print(len(aspects))
    model = SentimentUtilityLogisticModel(logger, ratings, num_aspects=len(aspects), num_factors=5,
                                          lambda_b = 0.05, lambda_pq = 0.05, lambda_z = 0.05, lambda_w = 0.05,
                                          gamma=2.0, iterations=5, alpha=0.5,
                                          l1 = False, l2 = True, mult = False)
    
    #model.sentiments_correlation()
    model.train_model()

    logger.info('Average Sentiments:\n%s'%str(list(zip(aspects, model.avg_sentiments))))
    model.pretty_save('readable_model.txt')
    model.predict_train()
    model.save('model_test_')

    modelnew = SentimentUtilityLogisticModel(logger, ratings,num_aspects=len(aspects), num_factors=5,
                                             lambda_b = 0.01, lambda_pq = 0.01, lambda_z = 0.08, lambda_w = 0.01,
                                             gamma=0.001,iterations=30, alpha=0.00)
    modelnew.load('model_test_')

    testset = [['user5','item3'],
               ['user1','item4'],
               ['user2','item4']
               ]

    user_input=input('enter a known user whose predicted experience you want to know:')
    item_input=input('enter a known item whose predicted experience user would have:')

    category=[]
    liquids=[]
    liquids={"DRINKS", "DRINKS_ALCOHOL", "DRINKS_ALCOHOL_BEER", "DRINKS_ALCOHOL_HARD", "DRINKS_ALCOHOL_LIGHT", "DRINKS_ALCOHOL_WINE", "DRINKS_NON-ALCOHOL_COLD", "DRINKS_NON-ALCOHOL_HOT"}
    food_variety=[]
    food_variety={"FOOD_FOOD", "FOOD_FOOD_BREAD", "FOOD_FOOD_CHEESE", "FOOD_FOOD_CHICKEN", "FOOD_FOOD_DESSERT", "FOOD_FOOD_DISH", "FOOD_FOOD_EGGS", "FOOD_FOOD_FRUIT", "FOOD_FOOD_MEAT", "FOOD_FOOD_MEAT_BACON", "FOOD_FOOD_MEAT_BEEF", "FOOD_FOOD_MEAT_BURGER", "FOOD_FOOD_MEAT_LAMB", "FOOD_FOOD_MEAT_PORK", "FOOD_FOOD_MEAT_RIB", "FOOD_FOOD_MEAT_STEAK", "FOOD_FOOD_MEAT_VEAL", "FOOD_FOOD_SALAD", "FOOD_FOOD_SAUCE", "FOOD_FOOD_SEAFOOD", "FOOD_FOOD_SEAFOOD_FISH", "FOOD_FOOD_SEAFOOD_SEA", "FOOD_FOOD_SIDE", "FOOD_FOOD_SIDE_PASTA", "FOOD_FOOD_SIDE_POTATO", "FOOD_FOOD_SIDE_RICE", "FOOD_FOOD_SIDE_VEGETABLES", "FOOD_FOOD_SOUP", "FOOD_FOOD_SUSHI"}
    food_timing={}
    food_timing={"FOOD_MEALTYPE_BREAKFAST", "FOOD_MEALTYPE_BRUNCH", "FOOD_MEALTYPE_DINNER", "FOOD_MEALTYPE_LUNCH", "FOOD_MEALTYPE_MAIN", "FOOD_MEALTYPE_START", "FOOD_PORTION", "FOOD_SELECTION"}
    restaurant_features=[]
    restaurant_features={"GENERAL", "PERSONAL", "RESTAURANT", "RESTAURANT_ATMOSPHERE", "RESTAURANT_CUSINE", "RESTAURANT_ENTERTAINMENT_MUSIC", "RESTAURANT_ENTERTAINMENT_SPORT", "RESTAURANT_INTERIOR", "RESTAURANT_INTERNET", "RESTAURANT_LOCATION", "RESTAURANT_MONEY", "RESTAURANT_PARKING", "SERVICE"}
    cat=[{'liquids','food_variety','food_timing','restaurant_features'}]
    category={}
    category={'liquids':{"DRINKS", "DRINKS_ALCOHOL", "DRINKS_ALCOHOL_BEER", "DRINKS_ALCOHOL_HARD", "DRINKS_ALCOHOL_LIGHT", "DRINKS_ALCOHOL_WINE", "DRINKS_NON-ALCOHOL_COLD", "DRINKS_NON-ALCOHOL_HOT"},'food_variety':{"FOOD_FOOD", "FOOD_FOOD_BREAD", "FOOD_FOOD_CHEESE", "FOOD_FOOD_CHICKEN", "FOOD_FOOD_DESSERT", "FOOD_FOOD_DISH", "FOOD_FOOD_EGGS", "FOOD_FOOD_FRUIT", "FOOD_FOOD_MEAT", "FOOD_FOOD_MEAT_BACON", "FOOD_FOOD_MEAT_BEEF", "FOOD_FOOD_MEAT_BURGER", "FOOD_FOOD_MEAT_LAMB", "FOOD_FOOD_MEAT_PORK", "FOOD_FOOD_MEAT_RIB", "FOOD_FOOD_MEAT_STEAK", "FOOD_FOOD_MEAT_VEAL", "FOOD_FOOD_SALAD", "FOOD_FOOD_SAUCE", "FOOD_FOOD_SEAFOOD", "FOOD_FOOD_SEAFOOD_FISH", "FOOD_FOOD_SEAFOOD_SEA", "FOOD_FOOD_SIDE", "FOOD_FOOD_SIDE_PASTA", "FOOD_FOOD_SIDE_POTATO", "FOOD_FOOD_SIDE_RICE", "FOOD_FOOD_SIDE_VEGETABLES", "FOOD_FOOD_SOUP", "FOOD_FOOD_SUSHI"},'food_timing':{"FOOD_MEALTYPE_BREAKFAST", "FOOD_MEALTYPE_BRUNCH", "FOOD_MEALTYPE_DINNER", "FOOD_MEALTYPE_LUNCH", "FOOD_MEALTYPE_MAIN", "FOOD_MEALTYPE_START", "FOOD_PORTION", "FOOD_SELECTION"},'restaurant_features':{"GENERAL", "PERSONAL", "RESTAURANT", "RESTAURANT_ATMOSPHERE", "RESTAURANT_CUSINE", "RESTAURANT_ENTERTAINMENT_MUSIC", "RESTAURANT_ENTERTAINMENT_SPORT", "RESTAURANT_INTERIOR", "RESTAURANT_INTERNET", "RESTAURANT_LOCATION", "RESTAURANT_MONEY", "RESTAURANT_PARKING", "SERVICE"}}
    print("Following are the categories of aspects");
    for n,i in enumerate(cat):
        print(i);
    #print("Enter the category name");
    user_cat=input("Enter the category name : ");
    print("Following are the aspects in this category");
    print(category[user_cat]);
    user_aspect=input("Enter the aspect : ");
    z=-1;
    for n,i in enumerate(aspects):
        if(user_aspect==i):
            z=n;
    #print(z);

    
    testset=[[user_input,item_input]]
    modelnew.predict_test(testset,'model_test.txt')
    #print('Enter the corrresponding no for following aspects')
    #for n,i in enumerate (aspects):
        #print (i + ' = ' + str(n))
    #aspect_input=input('Enter the no: ')
    modelnew.predict_best_rest_for_given_aspect(int(z))

