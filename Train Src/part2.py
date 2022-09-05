import pandas as pd
import numpy as np
import sklearn
import torch
import pickle

from utils import (
    get_loaders,
    load_data,
    ExcelDataset,
    FetchAugmentedDeath
)

from xg_boost import (
    xgb_regression
)

from linear_regression import (
    linear_regression
)

from bio_age_nn import(NeuralNetwork, bio_age_nn)
from DT import tree

from KNN import knn
from Linear import linear

from constants import *


if __name__ == "__main__":
    ct_data, clinical_data, outcomes_data,  = load_data(ALL_PATH, normalizeClinical=True)

    # for part 2
    
    # y = outcomes_data[:,]
    ailment_scores = torch.sum(outcomes_data[:, 1:], axis=1)
    for i in range(1,outcomes_data.size()[1]):
        if i%2!=0:
            ailment_scores+=outcomes_data[:,i]
        else:
            temp_score=outcomes_data[:,i]
            temp_score = (temp_score - torch.min(temp_score)) / \
        (torch.max(temp_score) - torch.min(temp_score))
            #ailment_scores+=(1/temp_score)
            ailment_scores+=temp_score
    ailment_scores = (ailment_scores - torch.min(ailment_scores)) / \
        (torch.max(ailment_scores) - torch.min(ailment_scores))


    print("Here")   

    y = FetchAugmentedDeath(clinical_data[:,0], ailment_scores, outcomes_data[:,0])

    x = torch.sum(outcomes_data[:, 1:], axis=1)
    x = torch.hstack((x.unsqueeze(1), ct_data))
    x = torch.hstack((x, clinical_data))

    x = torch.hstack((x,ailment_scores.unsqueeze(1)))

    
    df = pd.DataFrame(x)  # convert to a dataframe
    df.to_csv("X-for-Death.csv", index=False)

    df= pd.DataFrame(y)
    df.to_csv("Augmented-Y-for-death.csv", index=False)  

    part_2 = ExcelDataset(x, y)

    print("XGB")
    model = xgb_regression(x, y)
    pickle.dump(model, open(MODEL_PATH+"DeathCTandClinical.dat", "wb"))
    train_loader, test_loader, validation_loader = get_loaders(part_2)

    print("Linear using Torch")
    model = linear_regression(
         train_loader, test_loader,  x.shape[1], linear_output_dim)

    torch.save(model.state_dict(), MODEL_PATH + "lin_reg_model.pkl" )

    print("Linear using sklearn")
    model = linear(x, y)

    print("Neural Net")
    bio_age_model = bio_age_nn(
         train_loader, test_loader, x.shape[1], linear_output_dim)

    torch.save(bio_age_model.state_dict(), MODEL_PATH + "bio_age_model.pkl")

    print("DT")
    model = tree(x, y)

    print("KNN 5")
    model = knn(x, y, n=5)


    