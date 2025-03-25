
# importing the required libraries
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# define data values
# x = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.,  1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9])  # X-axis points
x = np.array([2,3,4,5,6,7,8,9,10,11,12,13,14,15,16])
baseline = [0.864799976348877] * 15
baseline_dp = [0.3321000039577484] * 15
HUSH = [0.84129998087883, 0.829366664091746, 0.8021750003099442, 0.809660005569458, 0.7894333402315775, 0.7955714208739144, 0.7820250019431114, 0.7785999973615011, 0.774700003862381, 0.7705181837081909, 0.7654749900102615, 0.7564769249695998, 0.751635708979198, 0.7495199998219808, 0.7463937476277351]
HUSH_dp = [0.7368499934673309, 0.7145666480064392, 0.6944500058889389, 0.6735199928283692, 0.6565333207448324, 0.619857132434845, 0.6352625042200089, 0.6196333368619283, 0.610589998960495, 0.6000909100879322, 0.5897916704416275, 0.5880000086931082, 0.5780857120241437, 0.5676066676775614, 0.5642062462866306]
SISA = [0.825300008058548, 0.7991666793823242, 0.7837000042200089, 0.7700000166893005, 0.7663833300272623, 0.7465000067438398, 0.7510999962687492, 0.7320444451438056, 0.7359600067138672, 0.7185909152030945, 0.716049998998642, 0.7150999995378348, 0.7072785667010716, 0.7021933317184448, 0.6977187506854534] # Y-axis points
plt.ylim([.2, 1])
plt.xlabel("shard count")
plt.ylabel("average accuracy")
# plt.xscale("log")
plt.plot(x, baseline, label = "baseline") 
plt.plot(x, baseline_dp, label = "baseline_dp") 
plt.plot(x, HUSH, label = "HUSH") 
plt.plot(x, HUSH_dp, label = "HUSH_dp") 
plt.plot(x, SISA, label = "SISA") 

plt.legend()
# plt.axhline(y=0.9186999797821045, color='red', linestyle='-')

# handles, labels = plt.gca().get_legend_handles_labels()
# HUSH_dp_legend = Line2D([0], [0], label='SISA', color='blue', linestyle='-')
# baseline_dp_legend = Line2D([0], [0], label='baseline_dp', color='red', linestyle='-')
# handles.extend([HUSH_dp_legend, baseline_dp_legend])

# plt.legend(handles=handles, loc='lower center', fancybox=True, ncol=5)


plt.savefig('vgg-accuracy-all') 