import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Sample housing data
data = {
    'Price': [250000, 350000, 450000, 300000, 400000],
    'SquareFeet': [1500, 1800, 2200, 1600, 2000],
    'Bedrooms': [2, 3, 4, 3, 4],
    'Age': [10, 5, 2, 15, 8]
}
df = pd.DataFrame(data)

# Generate pairplot
sns.pairplot(df[['Price', 'SquareFeet', 'Bedrooms', 'Age']])
plt.suptitle('Pairwise Relationships in Housing Data', y=1.02)
plt.show()