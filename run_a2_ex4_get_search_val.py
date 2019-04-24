import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    show_plots = False

    #
    json_files = [
        "assign2/a2_ex4_coarse_search.json",
        "assign2/a2_ex4_fine_search.json",
    ]
    # get the coarse search values
    for json_file in json_files:
        print(json_file)
        df = pd.read_json(json_file)
        df = df.drop("scores", axis="columns")
        df = df.sort_values(by=['mean_score'], ascending=False)
        print(df.iloc[:3])
        if show_plots:
            df.plot(x="lambda_", y="mean_score", kind="scatter", logx=True)
            plt.show()