import os

runs = os.listdir("out")

for run in runs:
    trials = os.listdir(f"out/{run}")
    for trial in trials:
        if "networks" in os.listdir(f"out/{run}/{trial}"):
            print(f"Found networks in {run}/{trial}")
            networks = os.listdir(f"out/{run}/{trial}/networks")
            network_image_filenames = [
                f"out/{run}/{trial}/networks/{network}" for network in networks
            ]
        else:
            print(f"No networks in {run}/{trial}")
            continue
