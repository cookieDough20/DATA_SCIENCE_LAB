import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

a = np.array([1,2,3,4,5])
print(a)

df = pd.DataFrame({"A":[1,2,3],"B":[4,5,6]})
print(df)

plt.plot(a)
plt.show()
