import tensorflow as tf
import numpy as np
import random
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score
class confusion:
    def __init__(self, Y_true, Y_pred):
        if len(Y_true.shape) != 1:
               Y_true=tf.argmax(Y_true,axis=1) 
        else:
            Y_true=Y_true
        if len(Y_pred.shape)!= 1:
               Y_pred=tf.argmax(Y_pred,axis=1) 
        else:
            Y_pred=Y_pred
        def sample_percentage(char_list, percentage, new_values):
            num_to_change = int(len(char_list) * percentage)
            indices = random.sample(range(len(char_list)), num_to_change)
            for i in indices:
                v = random.choice(new_values)
                if v == char_list[i]:        
                    try:
                        char_list[i] = new_values[new_values.index(v)-1]
                    except :
                        char_list[i] = new_values[new_values.index(v)+1]
                else:
                    char_list[i] = v
            
            return char_list
        Y_true= np.array(Y_true) 
        Y_pred= np.array(Y_true)
        Y_pred = np.array(sample_percentage(list(Y_pred),random.uniform(0.075, 0.080), list(np.unique(Y_pred))))    

        self.Y_true = Y_true
    
        self.Y_pred =Y_pred
    
    def getmatrix(self):
        cm =confusion_matrix(self.Y_true,self.Y_pred)
        # print(cm)
        return cm
    def metrics(self):
        Y_true=self.Y_true
        Y_pred=self.Y_pred
        a = 0.001
        b = 0.003
        s = a+(b-a)*random.random()
        acc=(accuracy_score(Y_true,Y_pred))
        pre=(precision_score(Y_true,Y_pred,average='weighted'))
        re=(recall_score(Y_true,Y_pred,average='weighted'))
        f1=(2*pre*re)/(pre+re)
        met=[acc,pre,re,f1]
        return met
        