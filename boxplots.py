import numpy as np
import matplotlib.pyplot as plt
import matplotlib

font = {'family' : 'normal',
        'size'   : 22}

matplotlib.rc('font', **font)
string1 = '/home/radillo/Git/GitHub/FinalProjectRice/heatmaps'
string3 = '.1/acc_array.npy'
a = []
for i in [2,3,4,5,6,7,8]:
    string2 = str(i)
    tmp = np.load(string1 + string2 + string3)
    a.append(tmp)

plt.figure()
plt.boxplot(a, 0, '')
plt.xticks([1,2,3,4,5,6,7],['2','3','4','5','6','7','8'])
plt.xlabel('patch size')
plt.ylabel('classif. acc.')
plt.title('Effect of patch size on acc.')
plt.show()