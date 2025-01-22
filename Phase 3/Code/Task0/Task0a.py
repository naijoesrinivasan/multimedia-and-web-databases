import pandas as pd
from scipy.io import savemat
from Utils import get_inherent_dim_even
import os

STORE_DIR = os.getcwd()+'/Store'

if __name__ == "__main__":
    features, feature_name  = get_inherent_dim_even()
    df = pd.DataFrame(columns=["Feature Name", "Dimentions"])
    
    savemat(STORE_DIR+'/inherent_even.mat', {"Inherent Dim Even": features})

    result = {"Feature Name": feature_name, "Dimentions": features.shape}
    df = pd.concat([df, pd.DataFrame([result])], ignore_index=True)
    df.to_csv(STORE_DIR+'/inherent_even.csv', index = False)
