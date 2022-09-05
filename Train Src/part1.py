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
from constants import *

if __name__ == "__main__":
    ct_data, clinical_data, outcomes_data,  = load_data(ALL_PATH)

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

    df = pd.DataFrame(clinical_data[:,0])  # convert to a dataframe
    df.to_csv("Pre augmented Y-for-Death.csv", index=False)

    y = FetchAugmentedDeath(clinical_data[:,0], ailment_scores, outcomes_data[:,0])

    x= torch.hstack((ct_data,ailment_scores.unsqueeze(1)))

    part_1 = ExcelDataset(x, y)

    train_loader, test_loader, validation_loader = get_loaders(part_1)

    health_scores = torch.sum(clinical_data[:, 5:10], axis=1)
    health_scores += clinical_data[:, 1]

    health_scores = (health_scores - torch.min(health_scores)) / \
        (torch.max(health_scores) - torch.min(health_scores))
    
   
    
    print("Linear using Torch")
    model = linear_regression(train_loader, test_loader,  ct_data.size()[1]+1, len(linear_predict) )

    torch.save(model.state_dict(), MODEL_PATH + "lin_reg_model.pkl" )

    print("Linear using sklearn")
    model = linear(x, y)

    print("Neural Net")
    bio_age_model = bio_age_nn(
        train_loader, test_loader, x.shape[1], linear_output_dim)

    print("DT")
    model = tree(x, y)

    print("KNN 4")
    model = knn(x, y, n=4)

    print("XGB")
    model = xgb_regression(x, y)
   
    pickle.dump(model, open("DeathFromCTOnly.dat", "wb"))
    