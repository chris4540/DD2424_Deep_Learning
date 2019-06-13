import pandas as pd
import sys

if __name__ == "__main__":
    json_file = sys.argv[1]
    json_name = json_file.split(".")[0]
    df = pd.read_json(json_file)
    df = df.set_index("weight_decay")
    df = df.sort_index()
    # print(df)

    ax = df.plot(kind='line', legend=True, marker='x', logx=True)
    ax.set_ylabel("Validation Acc.")
    ax.set_xlabel("lambda")
    fig = ax.get_figure()
    fig.savefig(json_name+'.png', bbox_inches='tight', dpi=200)
