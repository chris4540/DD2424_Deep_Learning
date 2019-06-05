import sys
import pandas as pd


if __name__ == "__main__":
    csvfile = sys.argv[1]
    csvfile_name = csvfile.split(".")[0]
    # print(csvfile)

    df = pd.read_csv(csvfile, index_col=0)

    ax = df.plot(kind='line', legend=False)
    ax.set_xlim([0, None])
    ax.set_ylabel("Smooth Cost")
    fig = ax.get_figure()
    png_fname = csvfile_name + ".png"
    fig.savefig(png_fname, bbox_inches='tight', dpi=100)
