import os
import pandas as pd
from scipy.io import savemat

from Utils import get_inherent_dim_label

STORE_DIR = os.getcwd()+'/Store'

if __name__ == "__main__":
    features_label = get_inherent_dim_label()
    
    df = pd.DataFrame(columns=["Label", "Feature Name", "Dimentions"])
    savemat(STORE_DIR+'/inherent_label.mat', features_label)

    for label, dim in features_label.items():
        result = {"Label": label, "Feature Name": dim[0], "Dimentions": dim[1].shape}
        df = pd.concat([df, pd.DataFrame([result])], ignore_index=True)
    
    df.to_csv(STORE_DIR+'/inherent_label.csv', index = False)