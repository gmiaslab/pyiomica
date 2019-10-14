
import pyiomica as pio

from pyiomica.extendedDataFrame import DataFrame
df = DataFrame(pio.np.array([[1,2,3],[4,2,6],[7,3,4],[2,2,8],[4,4,4]]).astype(float))
print(df)

df_Q_normed = df.quantileNormalize()
print(df)