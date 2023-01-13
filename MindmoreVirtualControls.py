# -*- coding: utf-8 -*-
"""
Script for generating virtual controls given parameters:
    - Age
    - Sex
    - Input method
    - Education


"""

import pandas as pd
import numpy as np
np.random.seed(0)

class MindmoreVirtualControl:
    
    def __init__(self, df):
        self.df = df
        
        self.df['eduInv'] = 1/self.df['edu']
        self.df['age2'] = self.df['age']**2
        self.df['ageEdu'] = self.df['age']*self.df['edu']
        self.df['sex'].replace({'men':1, 'women':0}, inplace=True)
        self.df['GENERAL_inputDevice'].replace({'mouse':0, 'trackpad':1,'touchscreen':2}, inplace=True)
        self.df['GENERAL_inputDevice'] = self.df['GENERAL_inputDevice'].astype('int')

    def CERAD_learning(self):
        self.df['CERAD_LEARNING_PREDICTED'] = 31.7942156995638 + (-0.0875905541536407 * self.df['age']) +\
            (self.df['eduInv'] * -75.9733008123953)

        SD = 3.24313630994147
        Length = len(self.df['CERAD_LEARNING'])
        
        self.df['CERAD_LEARNING'] = np.random.normal(self.df['CERAD_LEARNING_PREDICTED'], SD, Length)
        # self.df.drop('CERAD_LEARNING_PREDICTED', axis=1, inplace=True)
        
        self.df['CERAD_LEARNING'].where(self.df['CERAD_LEARNING'] <=30, 30, inplace=True)

        
        return self.df
    
    def CERAD_recall(self):
        
        self.df['CERAD_DELAYED_PREDICTED'] = 12.7904955190877 + (self.df['age']*-0.0501897879003986) +\
            (self.df['eduInv'] * -33.7554737721963) + (-0.810824961274903 * self.df['sex'])
            
        SD = 1.89431914995295
        Length = len(self.df['CERAD_DELAYED_PREDICTED'])               
        self.df['CERAD_DELAYED'] = np.random.normal(self.df['CERAD_DELAYED_PREDICTED'], SD, Length)
        # self.df.drop('CERAD_DELAYED_PREDICTED', axis=1, inplace=True)
        
        self.df['CERAD_DELAYED'].where(self.df['CERAD_DELAYED'] <=10, 10, inplace=True)

        
        return self.df
                
    def CORSI_FWD(self):
        
        self.df['CORSI_FWD_predicted'] = 5.82930531887294 + (self.df['age']*0.016789664249949) + \
            (self.df['age2']*-0.000465225653294214) + (self.df['edu']*0.0435293550465566) + \
                (self.df['sex']*0.272833659477026)
                        
        SD = 1.01018250326044
        Length = len(self.df['CORSI_FWD_predicted'])        
        self.df['CORSI_FWD_SPAN'] = np.random.normal(self.df['CORSI_FWD_predicted'], SD, Length)        
        self.df.drop('CORSI_FWD_predicted', axis=1, inplace=True)
        self.df['CORSI_FWD_SPAN'].where(self.df['CORSI_FWD_SPAN'] <=9, 9, inplace=True)

        
        return self.df
    
    def FAS(self):
        
        self.df['FAS_predicted'] = 31.9386200030632 + (self.df['edu']*0.92077653545719)
        
        SD = 13.5770025270257
        Length = len(self.df['FAS_predicted']) 
        self.df['FAS_INDEX'] = np.random.normal(self.df['FAS_predicted'], SD, Length)        

        return self.df

    def SDMT(self):
        
    
        def sdmt_standardizer(row):
            """
            Additional function in order to determine what model to standardize
            by
            
            
            """
            if row['GENERAL_inputDevice'] == 1:
                SDMT_PREDICTED = 68.931 + \
                (-4.441 * row['GENERAL_inputDevice']) + \
                (-0.477 * row['age']) + \
                    (1.656 * row['sex'])
                SD = 6.615                                
                
                SDMT_Score = np.random.normal(SDMT_PREDICTED, SD, 1) 
                return SDMT_Score[0]
                           
            if row['GENERAL_inputDevice'] == 0:
                SDMT_PREDICTED = 68.931 + \
                (-4.441 * row['GENERAL_inputDevice']) + \
                (-0.477 * row['age']) + \
                    (1.656 * row['sex'])
                SD = 6.615                                
                
                SDMT_Score = np.random.normal(SDMT_PREDICTED, SD, 1) 
                return SDMT_Score[0]
        
            elif row['GENERAL_inputDevice'] == 2:
                SDMT_PREDICTED = 56.0462056475881 + (-0.0615085632106278 * row['age']) + \
                    (-0.00492377037979746 * row['age2']) + (0.553681784665026 * row['edu'])
                SD = 6.88771206095076
                
                SDMT_Score = np.random.normal(SDMT_PREDICTED, SD, 1) 
                return SDMT_Score[0]

        self.df['SDMT_CORRECT'] = self.df.apply(lambda row: sdmt_standardizer(row), axis=1)
        df = self.df
        return df

    
    def Stroop_index_VC(self):
    
        def Stroop_standardizer(row):
            """
            Additional function in order to determine what model to standardize
            by
            
            
            """
            if row['GENERAL_inputDevice'] == 1:
                Stroop_PREDICTED = 20.994 + \
                (-1.929 * row['GENERAL_inputDevice']) + \
                (-0.147 * row['age'])
                SD = 2.395                                
                
                Stroop_Score = np.random.normal(Stroop_PREDICTED, SD, 1)            
                return Stroop_Score[0]
                           
            if row['GENERAL_inputDevice'] == 0:
                Stroop_PREDICTED = 20.994 + \
                (-1.929 * row['GENERAL_inputDevice']) + \
                (-0.147 * row['age'])
                SD = 2.395                                
                
                Stroop_Score = np.random.normal(Stroop_PREDICTED, SD, 1)            
                return Stroop_Score[0]
        
            elif row['GENERAL_inputDevice'] == 2:
                Stroop_PREDICTED = 18.2684200875865 + (-0.116902227050743 * row['age'])
                SD = 2.76981099623645
                
                Stroop_Score = np.random.normal(Stroop_PREDICTED, SD, 1)            
                return Stroop_Score[0]

        self.df['STROOP_INCONGRUENT_INDEX'] = self.df.apply(lambda row: Stroop_standardizer(row), axis=1)
        df = self.df
        return df

    
    
    def Stroop_inhibition_VC(self):
                
        def Stroop_standardizer_SD_picker(row):
            STROOP_INTERFERENCE_predicted = 64.748 + (row['age']*6.707)

            if (STROOP_INTERFERENCE_predicted < 296.210):
                
                SD = 199.392
                StroopInhibScore = np.random.normal(STROOP_INTERFERENCE_predicted, SD, 1)                                         
                return StroopInhibScore[0]
                    
            elif (STROOP_INTERFERENCE_predicted > 296.210 and 
                  STROOP_INTERFERENCE_predicted < 406.837):
                
                SD = 223.075
                StroopInhibScore = np.random.normal(STROOP_INTERFERENCE_predicted, SD, 1)                                         
                return StroopInhibScore[0]

            elif (STROOP_INTERFERENCE_predicted > 406.836 and 
                  STROOP_INTERFERENCE_predicted < 517.463):
    
                SD = 272.525
                StroopInhibScore = np.random.normal(STROOP_INTERFERENCE_predicted, SD, 1)                                         
                return StroopInhibScore[0]
    
            elif STROOP_INTERFERENCE_predicted >= 517.463:
    
                SD = 388.589
                StroopInhibScore = np.random.normal(STROOP_INTERFERENCE_predicted, SD, 1)                                         
                return StroopInhibScore[0]
                    
        self.df['STROOP_INTERFERENCE'] = self.df.apply(lambda row: Stroop_standardizer_SD_picker(row), axis=1)

        
        return self.df                    


        