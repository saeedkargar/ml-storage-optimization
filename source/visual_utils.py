#custom legend: https://matplotlib.org/3.5.0/gallery/text_labels_and_annotations/custom_legends.html
#colors in legend: https://matplotlib.org/3.5.0/users/prev_whats_new/dflt_style_changes.html
# resolution of figures: https://www.delftstack.com/howto/matplotlib/how-to-plot-and-save-a-graph-in-high-resolution/

import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from matplotlib.lines import Line2D


font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : 18}

plt.rc('font', **font)

def loadresults(filepath, key):
    with open(filepath) as json_file:
        data = json.load(json_file)

    temp = dict()
    for k in data.keys():
        temp[k] = dict()
        for l in data[k].keys():
            try:
                l_new = float("{:.2f}".format(float(l)))
            except:
                l_new = l
            temp[k][l_new] = data[k][l][key]


    data_df = pd.DataFrame.from_dict(temp,
                        orient='index').T

    return data_df


def visualize_root_size_effect(data, t, xl, yl):
    data.plot(figsize=(10, 5))

    plt.xlabel(xl)
    plt.ylabel(yl)
    plt.legend(loc=0)
    # plt.legend('', frameon=False)
    # plt.legend(loc=(1.04, 0), fancybox=True)
    # plt.get_legend().remove()

    plt.subplots_adjust(right=0.7)
    result_fig_name = f"{t}-{xl}-{yl}.png"
    path = os.path.split(os.getcwd())[0]
    figures_Path = path + "/figures/" + result_fig_name
    plt.savefig(figures_Path,dpi=600, bbox_inches="tight")
    plt.show()
    

def visualize_seg_count_effect(data, t, xl, yl):

    #get the dp benchmark value
    dp_benchmark_data = data.iloc[-1].iloc[0]
    #remove the dp benchmark row from the data 
    data = data.iloc[:-1]

    
    #get the benchmark value
    benchmark_data = data.iloc[-1].iloc[0]
    #remove the benchmark row from the data 
    data = data.iloc[:-1]

    data.plot(figsize=(8, 5))

    handles, labels = plt.gca().get_legend_handles_labels()

    
    
    # plt.text(1, benchmark_data,'Benchmark')
    plt.axhline(y=benchmark_data, color='red', linestyle='-')

    # plt.text(1, dp_benchmark_data,'Benchmark + DP')
    plt.axhline(y=dp_benchmark_data, color='red', linestyle='--')

    plt.xlabel(xl)
    plt.ylabel(yl)


    benchmark_line = Line2D([0], [0], label='Baseline', color='red', linestyle='-')
    benchmark_dp_line = Line2D([0], [0], label='Baseline + DP', color='red', linestyle='--')

    handles.extend([benchmark_line])
    handles.extend([benchmark_dp_line])

    plt.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.2),
          fancybox=True, ncol=5)

    plt.subplots_adjust(right=0.7)
    result_fig_name = f"{t}-{xl}-{yl}.png"
    path = os.path.split(os.getcwd())[0]
    figures_Path = path + "/figures/resnet/" + result_fig_name
    plt.savefig(figures_Path,dpi=600, bbox_inches="tight")

    # plt.show()