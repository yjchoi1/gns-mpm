# yc: Plot runout with varing phi to check \partial{rounout}/\partial{phi}

import numpy as np
import os
from matplotlib import pyplot as plt
import pickle

path = "/work2/08264/baagee/frontera/gns-mpm-data/gns-data/inverse/sand2d_frictions/"
cases = ["see_runout_gradient_over_phi_short_step380",
         "see_runout_gradient_over_phi_tall_step380"]

# Dict to store data
runout_dict = {
    "see_runout_gradient_over_phi_short_step380": {
        "frictions": [],
        "final_runouts": []
    },
    "see_runout_gradient_over_phi_tall_step380": {
        "frictions": [],
        "final_runouts": []
    }
}

# Open `.pkl` and store data to dict
for case in cases:
    file = open(os.path.join(path, case, "record.pkl"), 'rb')
    data = pickle.load(file)
    for d in data:
        runout_dict[case]["frictions"].append(d["phi"])
        runout_dict[case]["final_runouts"].append(d["final_runout"])

# Plot
for case, value in runout_dict.items():
    fig, ax = plt.subplots()
    ax.plot(value["frictions"], value["final_runouts"])
    ax.set_xlabel("Friction angle")
    ax.set_ylabel("Runout")
    plt.savefig(os.path.join(path, case, "runout_over_phi.png"))




