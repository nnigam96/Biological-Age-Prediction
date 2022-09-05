from statistics import mean
import pandas as pd
import numpy as np
import sklearn
import torch
from xg_boost import (
    xgb_regression
)
import pickle

from Linear import linear

from linear_regression import (
    linear_regression
)

from bio_age_nn import(NeuralNetwork, bio_age_nn)
from DT import tree

from KNN import knn
from Linear import linear


from utils import (
    get_loaders,
    load_data,
    ExcelDataset
)

from linear_regression import (
    linear_regression,
    LinearRegression
)
from bio_age_nn import(NeuralNetwork, bio_age_nn)
from constants import *

if __name__ == "__main__":

    ct_data, clinical_data, outcomes_data, = load_data(ALL_PATH)

    # for part 2
    y = pd.read_csv("Augmented-Y-for-death.csv")
    y=y.to_numpy()
    # y = outcomes_data[:,]
    x = pd.read_csv("X-for-Death.csv")
    x=x.to_numpy()
    
    # ailment_scores
    
    health_scores = torch.sum(clinical_data[:, 5:10], axis=1)
    health_scores += clinical_data[:, 1]
    health_scores = (health_scores - torch.min(health_scores)) / \
        (torch.max(health_scores) - torch.min(health_scores))
    

    Lifestyle_score = x[:,-1]+health_scores.numpy()
    x[:,-1]=Lifestyle_score
    model = linear(x, y)
    
    death = outcomes_data[:,DEATH_COL].numpy()

    for i in range(len(death)):
        if death[i] == 0:
            y_hat = model.predict(x[i].reshape(1,-1))
            death[i] = y_hat

    print(death.shape)

    # this will be x1
    lifespan = [float(sum(x))
                for x in zip(clinical_data[:, AGE_COL], death[:]/365)]

    # get Age col
    y = clinical_data[:, [AGE_COL]]

    df = pd.DataFrame(lifespan)  # convert to a dataframe
    df.to_csv("Lifespan.csv", index=False)  # save to file

    norm_ct_df, norm_clinical_df, norm_outcomes_df, = load_data(ALL_PATH)
    # print(outcomes_df.columns)
    x = torch.hstack((torch.tensor(lifespan).unsqueeze(1),
                     torch.tensor(Lifestyle_score).unsqueeze(1)))

    
    #x = torch.hstack((x, health_scores.unsqueeze(1)))

    df = pd.DataFrame(x)  # convert to a dataframe
    df.to_csv("X-for-Bio-NN.csv", index=False)  # save to file

    #Categorize y based on mean and std:
    mean_score=torch.mean(torch.tensor(Lifestyle_score))
    std_score=torch.std(torch.tensor(Lifestyle_score))

    y = 80 * torch.ones(health_scores.unsqueeze(1).shape)

    
    for i,temp in enumerate(y):
        if Lifestyle_score[i]>mean_score+std_score:
            continue
        elif Lifestyle_score[i]<mean_score-std_score:
            y[i]=clinical_data[i, [AGE_COL]]
        else:
            y[i]=clinical_data[i,[AGE_COL]]*80/lifespan[i]
    
    x= torch.hstack((x,torch.ones(len(x)).unsqueeze(1)))

    x=x.float()
    y=y.float()

    part_4 = ExcelDataset(x, y)


    
    train_loader, test_loader, validation_loader = get_loaders(part_4)
    bio_age_model = bio_age_nn(
        train_loader, test_loader, x.shape[1], linear_output_dim)
    torch.save(bio_age_model.state_dict(), MODEL_PATH + "bio_age_model.pkl")

  

    bio_age = torch.zeros((len(lifespan), 1))
    bio_age_model.eval()
    for i in range(len(bio_age)):
        bio_age[i] = bio_age_model(x[i].view(1,3))
   

    df=pd.DataFrame(bio_age.detach().numpy())
    df.to_csv("BioAge.csv", index=False)  # save to file

    print("DT")
    model = tree(x, y)

    print("XGB")
    model = xgb_regression(x, y)
    bio_age=model.predict(x)
    
    df=pd.DataFrame(y)
    df.to_csv("Y-For-BioAge.csv", index=False)
    df=pd.DataFrame(bio_age)
    df.to_csv("BioAge.csv", index=False)  # save to file
    pickle.dump(model, open("BioAgePred.dat", "wb"))


    
    #print("Linear using Torch")
    #model = linear_regression(
         #train_loader, test_loader,  x.shape[1], linear_output_dim)

    print("Linear using sklearn")
    model = linear(x, y)

   
    print("Neural Net")
    bio_age_model = bio_age_nn(
         train_loader, test_loader, x.shape[1], linear_output_dim)
    
    print("KNN 5")
    model = knn(x, y, n=5)

