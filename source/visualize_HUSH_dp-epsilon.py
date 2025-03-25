
# importing the required libraries
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# define data values
# x = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.,  1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9])  # X-axis points
x = np.array([25313.692669217704, 177.96901194282282, 41.686915704583214, 16.56045965981191, 7.966869489572056, 4.438632736391091, 2.738616560643572, 1.9485683174360546, 1.3686550679119434, 1.0439139555758215, 0.8400723429643507, 0.6677450233653279, 0.5631726490328056, 0.4836580492381367, 0.41486592325272653, 0.3647121093276581, 0.32531665700959556, 0.2935428901026048, 0.26635028032310915])
y = np.array([0.9294000029563904, 0.9322400093078613, 0.9432600140571594, 0.9430199980735778, 0.9365199923515319, 0.9486400127410889, 0.9352400064468384, 0.9426399946212769, 0.9284199953079224, 0.938919997215271, 0.9377400040626526, 0.9258400082588196, 0.9328400015830993, 0.9415199995040894, 0.9355199933052063, 0.9386799931526184, 0.9279799818992615, 0.9277200102806091, 0.9280800104141236])  # Y-axis points
plt.ylim([.83, .95])
plt.xlabel("epsilon")
plt.ylabel("average accuracy")
plt.xscale("log")
plt.plot(x, y)  # Plot the chart

plt.axhline(y=0.947759997844696, color='red', linestyle='-')

handles, labels = plt.gca().get_legend_handles_labels()
HUSH_dp_legend = Line2D([0], [0], label='HUSH_dp', color='blue', linestyle='-')
SISA_legend = Line2D([0], [0], label='SISA', color='red', linestyle='-')
handles.extend([HUSH_dp_legend, SISA_legend])

plt.legend(handles=handles, loc='lower center', fancybox=True, ncol=5)


plt.savefig('5segments') 