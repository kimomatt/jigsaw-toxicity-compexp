# script to inspect the inputs that trigger a respective neuron to activate at a level within a certain threshold

import numpy as np
import pandas as pd 

# will just hardcode for now, can be made more general later 

acts = np.load("/workspace/compexp_outputs_full_mean_pool/val_activations.npy")
meta = pd.read_csv("/workspace/compexp_outputs_full_mean_pool/val_metadata.csv")

# roughly what each of these look like:
 
# val_activations.npy
# [
#   [ 0.12, -0.44, 1.87, ... ],
#   [ 0.03,  0.91, 0.15, ... ],
#   ...
# ]

# input,labels
# "This comment text ...","[0, 1, 0, ...]"
# "Another comment ...","[1, 0, 0, ...]"
# ...

# Top explanations for neuron 277 for interval (0.37109375, 1.390625):

neuron = 277
low, high = 0.37109375, 1.390625

mask = (acts[:, neuron] >= low) & (acts[:, neuron] <= high)

# .loc[mask] keeps only the rows where the mask is true, meta and acts are aligned so the mask can be applied to meta to get the corresponding input texts and labels for the activations that fall within the specified range for neuron 277. The .copy() method is used to create a copy of the filtered DataFrame, which can be useful if we want to modify it later without affecting the original meta DataFrame.
rows = meta.loc[mask].copy()

# adding column for activations so that i can see the inputs next to he activation values
rows["activation"] = acts[mask, neuron]

# sorting from highest activation value to lowest 
print(rows[["activation", "input"]].sort_values("activation", ascending=False).to_string(index=False))