# %%
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
sns.set_theme(context='paper', font='serif', palette='viridis', style='whitegrid')
# %%
# if __name__ == '__main__':
epsilon = [1.0, 3.0, 5.0, 7.0, 10.0]
acc = [62.6, 70.0, 71.5, 73.5, 73.8]
# %%
plt.plot(epsilon, acc, linestyle='--', marker='o', label='SmoothNet')
plt.scatter([7.53], [66.2], label='Papernot et al., 2020', color='orange', marker='v')
plt.scatter(7.5, 71.7, label='Klause et al., 2022', color='green', marker='v')
plt.scatter(7.42, 70.1, label='Dörmann et al., 2021', color='red', marker='v')
plt.scatter(8.0, 81.4, label='De et al, 2022', color='pink', marker='v')
plt.ylim(40, 90)
plt.xlabel("$\\varepsilon$")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("paretofront_cifar.png", dpi=300)
plt.show()

# %%
