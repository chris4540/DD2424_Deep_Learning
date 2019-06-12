import sys
import pandas as pd
import matplotlib.ticker as plticker

if __name__ == "__main__":
    csvfile = sys.argv[1]
    csvfile_name = csvfile.split(".")[0]

    df = pd.read_csv(csvfile)


    ax = df.plot(kind='line', legend=True, marker='x')
    loc = plticker.MultipleLocator(base=5) # this locator puts ticks at regular intervals
    ax.xaxis.set_major_locator(loc)
    ax.set_ylabel("Loss")
    ax.set_xlabel("Epoch")
    fig = ax.get_figure()
    png_fname = csvfile_name + ".png"
    fig.savefig(png_fname, bbox_inches='tight', dpi=200)
