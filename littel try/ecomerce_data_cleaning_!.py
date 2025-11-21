import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


data = {
    'Order_ID': [101, 102, 103, 104, 105, 106, 106, 108, 109],
    'Coffee_Type': ['Latte', 'Espresso', 'Cappuccino', 'latte', 'Espresso', 'Mocha', 'Mocha', None, 'Cappuccino'],
    'Price': [4.50, 3.00, 4.00, 4.50, '3.00', 5.00, 5.00, 3.50, 4.00],
    'Quantity': [1, 2, 1, 3, 1, 2, 2, 1, 1],
    'Date': ['2023-10-01', '2023-10-01', '2023-10-02', '2023-10-02', '2023-10-03', '2023-10-03', '2023-10-03', '2023-10-04', '2023-10-05']
}
df = pd.DataFrame(data)

most_frequence = df['Coffee_Type'].mode()[0]


df['Date'] = pd.to_datetime(df['Date'])
# df['Price'] = pd.to_numeric(df['Price'])
df['Price'] = df['Price'].astype(float)
df['Coffee_Type'] = df['Coffee_Type'].fillna(most_frequence)
df['Coffee_Type'] = df['Coffee_Type'].str.capitalize()
df = df.drop_duplicates()
df = df.reset_index(drop=True)

print(F"Full Dataset:\n{df}")
print(F"\nData first 5 rows:\n{df.head()}")
print(F"\nData last 5 rows:\n{df.tail()}\n")
print(f"\nData info: {df.info()}")
print(F"\nData describe:\n{df.describe()}")
print(F"\n\n\nFinal Dataset:\n{df}")


value_count = df['Coffee_Type'].value_counts()


fig, ax= plt.subplots(1, 2, figsize=(13,7))
ax[0].bar(value_count.index, value_count.values, color=['#C0D731','#E8B55C','#B74914','#5A713B'], label=['Cappuccino','Latte','Espresso','Mocha'])
ax[0].legend(loc=1, shadow=True)
ax[0].set_xlabel("Coffee Type")
ax[0].set_ylabel("Popularity")
ax[0].set_title("Popularity Of Coffes")
ax[1].pie(value_count, labels=value_count.index, colors=['#C0D731','#E8B55C','#B74914','#5A713B'], autopct="%1.1f%%", startangle=90)
ax[1].legend(loc=2, shadow=True, fontsize=9,)
ax[1].set_title("Popularity Of Coffes")
plt.tight_layout()
plt.show()

new_df = df.groupby('Date')['Quantity'].sum()
print(new_df)
plt.subplots(figsize=(10,5))
plt.plot(new_df.index,new_df.values, label="trading", color="#00A2FF", marker="x")
plt.title("Trending")
plt.xlabel("Date")
plt.ylabel("Quantity")
plt.grid(linewidth=0.6, linestyle=":")
plt.tight_layout()
plt.show()

df["Revenue"] = df['Price'] * df['Quantity']
Revenue = df.groupby("Date")['Revenue'].sum()

plt.subplots(figsize=(10,5))
plt.bar(Revenue.index, Revenue.values, label="Revenue", color=['#C0D731','#E8B55C','#B74914','#5A713B',"#5E6851"])
plt.xlabel("Product")
plt.ylabel("Per Product revenue")
plt.title("Revenue")
plt.grid(linewidth=0.6, linestyle=":")
plt.legend()
plt.tight_layout()
plt.show()