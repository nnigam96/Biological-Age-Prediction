import pickle
import pandas as pd
import numpy as np

X= pd.read_csv("X-for-Death.csv")
X_new=X.iloc[0]
X_new=X_new.to_numpy()
X_new=X_new.reshape(1,-1)

objectRep = open("models\BioAgePred.dat","rb")
BioAgePred= pickle.load(objectRep)

objectRep = open("models\DeathFromCTOnly.dat","rb")
DeathfromCTOnly=pickle.load(objectRep)

objectRep = open("models\DeathCTandClinical.dat","rb")
DeathCTandClinical=pickle.load(objectRep)

print("Here")

death_pred= DeathCTandClinical.predict(X_new)
print("Prediction for number of days since CT for demise:",death_pred)

# Assume age is 37 and ailment score is 0.17(Healthy)
age=37
Lifespan = age+death_pred
ailment_score=0.17

X2=[Lifespan,ailment_score,1]
X2=np.array(X2)
X2=X2.reshape(1,-1)
BioAge=BioAgePred.predict(X2)
print("Bio age for healthy 37 yr old:",BioAge)

age=57
Lifespan = age+death_pred
ailment_score=0.89

X2=[Lifespan,ailment_score,1]
X2=np.array(X2)
X2=X2.reshape(1,-1)
BioAge=BioAgePred.predict(X2)
print("Bio age for extremely unhealthy 57 yr old:",BioAge)

