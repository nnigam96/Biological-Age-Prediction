from turtle import color
import matplotlib.pyplot as plt
import numpy as np
import pandas
import pandas as pd
from constants import *
from utils import (
    get_loaders,
    load_data,
    ExcelDataset
)

_, clinicaldata, _ = load_data(ALL_PATH)


Lifespan = pd.read_csv("X-For-Bio-NN.csv")
Lifestyle_score=Lifespan['1'].to_numpy()
Lifespan = Lifespan['0'].to_numpy().reshape(len(Lifespan))

mean_score=np.mean(Lifestyle_score)
std_score=np.std(Lifestyle_score)
z=pd.read_csv("BioAge.csv")

z=z.to_numpy().reshape(len(z))

col=[]
for i,temp in enumerate(z):
        if Lifestyle_score[i]>mean_score+std_score:
            col.append('r')
            continue
        elif Lifestyle_score[i]<mean_score-std_score:
            col.append('g')
        else:
            col.append('y')


y=clinicaldata[:,[AGE_COL]]
y=y.squeeze()

Bio_age_y=pd.read_csv("Y-For-BioAge.csv")
cmap = plt.get_cmap('viridis', 3)
cmap.set_under('red')


plt.scatter(y.numpy(),z,color='g')
plt.plot(y.numpy(),y.numpy(), color='b')

plt.xlabel("Chronological Age")
plt.ylabel("Biological Ages")
plt.title("Bio Age vs Chronological Age")
plt.show()

print("Fin")