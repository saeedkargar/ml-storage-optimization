
# add #%% to run cell
import sys
from visual_utils import *
from utils import alert

# USAGE: python3 visualize.py <here/is/the/path/to/file.json> <key to plot> <experiment-type>
# key to plot: 
    # "max_acc"
    # "min_acc"
    # "avg_acc"
    # "max_loss"
    # "min_loss"
    # "avg_loss"
    # "training_latency"
    # "retraining_latency"
    # "prediction_latency"
# experiment type:
    # segment-count
    # root-size

if len(sys.argv) < 2:
    print('USAGE: python3 visualize.py <here/is/the/path/to/file.json> <key to plot> <experiment type')
    exit()

filepath = sys.argv[1]
key = sys.argv[2]
exp_type = sys.argv[3]

data = loadresults(filepath, key)

if exp_type == 'segment-count':
    # figure 1: effect of increasing number of segments on initial training time
    if key == 'training_latency':
        title = "figure 1: effect of increasing number of segments on initial training time"
        xlabel = 'number of segments'
        ylabel = 'initial training latency (s)'

    # figure 2: effect of increasing number of segments on retraining time
    elif key == 'retraining_latency':
        title = "figure 2: effect of increasing number of segments on retraining time"
        xlabel = 'number of segments'
        ylabel = 'retraining latency (s)'

    # figure 3: effect of increasing number of segments on average accuracy
    elif key == 'max_acc':
        title = "figure 3.a: effect of increasing number of segments on max accuracy"
        xlabel = 'number of segments'
        ylabel = 'max accuracy'

    elif key == 'min_acc':
        title = "figure 3.c: effect of increasing number of segments on min accuracy"
        xlabel = 'number of segments'
        ylabel = 'min accuracy'

    elif key == 'avg_acc':
        title = "figure 3.b: effect of increasing number of segments on avg accuracy"
        xlabel = 'number of segments'
        ylabel = 'avg accuracy'

    visualize_seg_count_effect(data, title, xlabel, ylabel)

elif exp_type == 'root-size':
    # Figure 4: effect of changing root size on initial training time
    if key == 'training_latency':
        title = "figure 4: effect of changing root size on initial training time"
        xlabel = 'root size (%)'
        ylabel = 'initial training time'


    # Figure 5: effect of changing root size on retraining time
    elif key == 'retraining_latency':
        title = "Figure 5: effect of changing root size on retraining time"
        xlabel = 'root size (%)'
        ylabel = 'retraining time'


    # Figure 6: effect of changing root size on accuracy (max, avg, min)
    elif key == 'max_acc':
        title = "figure 6.a: effect of changing root size on max accuracy"
        xlabel = 'root size (%)'
        ylabel = 'max accuracy'

    elif key == 'min_acc':
        title = "figure 6.c: effect of changing root size on min accuracy"
        xlabel = 'root size (%)'
        ylabel = 'min accuracy'

    elif key == 'avg_acc':
        title = "figure 6.b: effect of changing root size on avg accuracy"
        xlabel = 'root size (%)'
        ylabel = 'avg accuracy'

    visualize_root_size_effect(data, title, xlabel, ylabel)

elif exp_type == 'prepare':
    print(data)
