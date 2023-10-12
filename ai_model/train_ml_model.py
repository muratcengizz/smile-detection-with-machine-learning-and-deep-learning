import os
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

class TrainMachineLearningModel:
    def __init__(self):
        self.ROOT_DIR = "C:\\Users\\murat\\Documents\\tensorflow\\body_language_decoder"
        self.dataset = f"{os.path.join(self.ROOT_DIR, 'dataset')}\\coords.csv"
        self.model = f"{self.ROOT_DIR}\\model\\body_language.pkl"
    
    def readInCollectedData(self):
        df = pd.read_csv(self.dataset)
        return df
    
    def trainTestSplit(self):
        dataset = self.readInCollectedData()
        
        X = dataset.drop(labels="class", axis=1)
        y = dataset["class"]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)
        return X_train, X_test, y_train, y_test
    
    def trainModel(self):
        pipelines = {
            'lr': make_pipeline(StandardScaler(), LogisticRegression()),
            'rc': make_pipeline(StandardScaler(), RidgeClassifier()),
            'rf': make_pipeline(StandardScaler(), RandomForestClassifier()),
            'gb': make_pipeline(StandardScaler(), GradientBoostingClassifier())
        }
        
        X_train, X_test, y_train, y_test = self.trainTestSplit()
        
        fit_models = {}
        for algorithm, pipeline in pipelines.items():
            model = pipeline.fit(X_train, y_train)
            fit_models[algorithm] = model
            
        fit_models
        
        with open(self.model, 'wb') as f:
            pickle.dump(fit_models['rf'], f)
        
    def evaluateAndSerializeModel(self):
        with open(self.model, "rb") as f:
            model = pickle.load(f)
        
        X_train, X_test, y_train, y_test = self.trainTestSplit()
        y_pred = model.predict(X_test)
        
        print("Ridge Classification Accuracy: ", accuracy_score(y_test, y_pred))
        
    def runAllProcess(self):
        print("Train-Test setlerine ayırma işlemi başlıyor...")
        try:
            X_train, X_test, y_train, y_test = self.trainTestSplit()
            print("Train-Test setlerine ayırma işlemi başarıyla bitti.")
        except Exception as e:
            print(f"trainTestSplit Fonksiyonunda hata!\n{e}")
        
        print("Model Eğitme işlemi başlıyor...")
        try:
            self.trainModel()
            print("Model eğitme işlemi başarıyla bitti")
        except Exception as e:
            print(f"trainModel fonksiyonunda hata!\n{e}")
        
        print("Modeli değerlendirme işlemi başlıyor...")
        try:
            self.evaluateAndSerializeModel()
            print("Modeli Değerlendirme işlemi bitti.")
        except Exception as e:
            print(f"evaluateAndSerializeModel fonksiyonunda hata!\n{e}")
        
    
p1 = TrainMachineLearningModel()
p1.runAllProcess()