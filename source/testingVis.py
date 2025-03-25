#%%
import sys
from visual_utils import *
import numpy as np

# %%

models = ['mnist-basic', 'vgg', 'resnet']
keys = ['retraining_latency', 'avg_acc']
figs = list()
for model in models:
    for key in keys:
        filepath = f"/Users/samaagazzaz/Desktop/UCSC_Courses/IPPML/experiments/experiments/results/root-size/{model}/root-size-4-{model}-HUSH.json"
        exp_type = 'root-size'

        figs.append(loadresults(filepath, key))

# %%
if exp_type == 'root-size':
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

# %%
fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(13, 9))

l1,=ax[0,0].plot(figs[0], color='r')
ax[0,1].plot(figs[1], color='r')
l2,=ax[1,0].plot(figs[2], color='b')
ax[1,1].plot(figs[3], color='b')
l3,=ax[2,0].plot(figs[4], color='g')
ax[2,1].plot(figs[5], color='g')

ax[1,0].set_ylabel('retraining latency (s)')
ax[1,1].set_ylabel('average accuracy')
ax[2,0].set_xlabel('trunk size (%)')
ax[2,1].set_xlabel('trunk size (%)')

plt.legend([l1, l2, l3],["MNIST-basic", "VGG", "RESNET"], loc='upper left')
plt.show() 
# %%
