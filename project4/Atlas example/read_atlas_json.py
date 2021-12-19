import json
import numpy as np

with open("atlas_params.json") as f:
    data = json.load(f)

print(f"The images are {data['Input files']}")
corrs = data["Correspondences"]
corrs_array = np.array(corrs)
print(f"There are {corrs_array.shape[0]} sets of correspondences across {corrs_array.shape[1]} images in a  {corrs_array.shape[2]}-dimensional domain")
print(f"The output file is {data['Output file']}")
