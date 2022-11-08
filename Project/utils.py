import pickle,json
import numpy as np
try:
    import config
except:
    pass

class wine():
    def __init__(self,alcohol,malic_acid,ash,alcalinity_of_ash,magnesium,total_phenols,flavanoids,nonflavanoid_phenols,proanthocyanins,color_intensity,hue,Diluted_wine,proline):
        self.alcohol=alcohol
        self.malic_acid=malic_acid
        self.ash=ash
        self.alcalinity_of_ash=alcalinity_of_ash
        self.magnesium=magnesium
        self.total_phenols=total_phenols
        self.flavanoids=flavanoids
        self.nonflavanoids_phenols=nonflavanoid_phenols
        self.proanthocyanins=proanthocyanins
        self.color_intensity=color_intensity
        self.hue=hue
        self.Diluted_wine=Diluted_wine
        self.proline=proline

    def load_model(self):
        try:
            with open(config.MODEL_PATH,"rb") as f:
                self.log_model=pickle.load(f)

            with open(config.JSON_PATH,"r") as f:
                self.json_data=json.load(f)
        except:
            with open('logistic_model.pkl',"rb") as f:
                self.log_model=pickle.load(f)

            with open('json_data.json',"r") as f:
                self.json_data=json.load(f)

    def prediction(self):
        self.load_model()
        array=np.array([self.alcohol,self.malic_acid,self.ash,self.alcalinity_of_ash,self.magnesium,self.total_phenols,self.flavanoids,self.nonflavanoids_phenols,self.proanthocyanins,self.color_intensity,self.hue,self.Diluted_wine,self.proline])
        print(array)
        prediction=self.log_model.predict([array])[0]
        print(prediction)
        return prediction
if __name__=="__main__":
    result=wine(15.23,1.61,2.78,16.23,123.00,2.36,2.78,0.18,2.71,4.78,1.71,2.89,1021.89)
    result.prediction()
