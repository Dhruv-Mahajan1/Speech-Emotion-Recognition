# import IPython
from pyexpat import model
import numpy as np
import wave
import warnings
import pandas as pd
import os
import soundfile
import librosa
import glob
import pickle
import lightgbm
import joblib
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score as adr
import pandas as pd


class SpeechEmotionRecognition:
  def __init__(self,filename,n_mfcc):
    self.filename=filename
    self.n_mfcc=n_mfcc

  def ExtractFeatures(self):
    self.y, self.samplingrate = librosa.load(self.filename, duration=3, offset=0.5)
    self.mfcc = np.mean(librosa.feature.mfcc(y=self.y, sr=self.samplingrate, n_mfcc=self.n_mfcc).T, axis=0)
    self.mfcc=np.array(self.mfcc)

Dataset=pd.read_csv('X - X.csv')
X= Dataset.drop(labels=['label'],axis=1)
Y=Dataset.iloc[:,40]
X=X.to_numpy()
Y=Y.to_numpy()
model = lightgbm.LGBMClassifier(num_leaves=30,subsample=0.8,max_depth=5,n_estimators=400,learning_rate=0.05,reg_lambda=1.2,reg_alpha=1.2)
model.fit(X,Y)

filename='03-01-01-01-01-01-01.wav'
n_mfcc=40
obj=SpeechEmotionRecognition(filename,n_mfcc)
obj.ExtractFeatures()

z=obj.mfcc
z.shape
ypr=model.predict([z])
print(ypr)
pickle.dump(model, open('modellgbm.pkl','wb'))

