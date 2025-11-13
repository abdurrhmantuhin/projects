import pandas as pd
import numpy as np
from datetime import datetime, timedelta

print("="*70)
print("E-COMMERCE SALES ANALYSIS PROJECT")
print("="*70)

np.random.seed(42)
n_customer = 100
n_order = 500
customer_name = np.random.choice(['Ahmed', 'Fatima', 'Ali', 'Zainab', 'Hassan', 'Aisha', 'Omar', 'Leila', 'Karim', 'Noor'], n_customer)

customers_data = {
    'customer_id':[f"cust{i:04d}" for i in range(1, n_customer+1)],
    'customer_name':customer_name,
    'email':[f"customer{i}@email.com" for i in range(1, n_customer+1)],
    'city':np.random.choice(['Dhaka', 'Chittagong', 'Sylhet', 'Khulna', 'Rajshahi'], n_customer),
    'signup_date':[datetime.now() - timedelta(days=np.random.randint(1, 365)) for _ in range(n_customer)]
}
customers_df = pd.DataFrame(customers_data)

order_data = {
    'order_id':[f"ord{i:06d}" for i in range(1, n_order+1)],
    'customer_id':np.random.choice(customers_data['customer_id'], n_order),
    'product_name':np.random.choice(['Laptop', 'Phone', 'Tablet', 'Headphones', 'Monitor', 'Keyboard', 'Mouse', 'USB Cable'], n_order),
    'quantity':np.random.randint(1,5,n_order),
    'price_per_unit':np.random.choice([500, 1000, 1500, 2000, 3000, 5000, 8000, 15000], n_order),
    'order_data':[datetime.now() - timedelta(days=np.random.randint(1,90)) for _ in range(n_order)],
    'payment_method': np.random.choice(['Cash', 'Card', 'Mobile Banking', 'Bkash'], n_order),
    'order_status': np.random.choice(['Completed', 'Pending', 'Cancelled'], n_order, p=[0.7, 0.2, 0.1])
}
order_df = pd.DataFrame(order_data)

products_data = {
    'product_name': ['Laptop', 'Phone', 'Tablet', 'Headphones', 'Monitor', 'Keyboard', 'Mouse', 'USB Cable'],
    'stock_quantity': [50, 200, 150, 300, 75, 400, 500, 1000],
    'category': ['Electronics', 'Electronics', 'Electronics', 'Accessories', 'Electronics', 'Accessories', 'Accessories', 'Accessories'],
    'supplier': ['Dell', 'Samsung', 'Apple', 'Sony', 'LG', 'Logitech', 'Razer', 'Generic']
}

products_df = pd.DataFrame(products_data)

print("\n✅ Data Created Successfully!")
print(f"\nCustomers: {len(customers_df)} | Orders: {len(order_df)} | Products: {len(products_df)}")


print("\n" + "="*70)
print("PART 2: DATA EXPLORATION")
print("="*70)

print("\n--- Customers DataFrame Info ---")
print(customers_df.head())
print(f"\nshape: {customers_df.shape}")
print(f"\ncolumns: {',  '.join(customers_df.columns.tolist())}")
print(f"\ndatatype :{customers_df.dtypes}")
print(f"\ndescribe:\n{customers_df.info()}")

print("\n--- Orders DataFrame Info ---")
print(f"\nOrder head view:\n{order_df.head()}")
print(f"\nShape:{order_df.shape}")

print("\n--- Products DataFrame Info ---")
print(products_df.head())
print(f"\nShape: {products_df.shape}")



print("\n" + "="*70)
print("PART 3: DATA FILTERING & SELECTION")
print("="*70)

high_value_orders = order_df[(order_df['price_per_unit'] * order_df['quantity'] > 15000) & (order_df['order_status'] == 'Completed')]
print(f"\n✓ High Value Completed Orders (>15000 TK): {len(high_value_orders)}")
print(high_value_orders[['order_id', 'product_name', 'quantity', 'price_per_unit', 'order_status']].head())

dhaka_customers = customers_df[customers_df['city'] == 'Dhaka']
print(f"\nDhaka customer:{len(dhaka_customers)}")

low_stock = products_df[(products_df['stock_quantity'] < 100) & (products_df['category'] == 'Electronics')]
print(f"\nLow stock Electronics:\n{low_stock}")




print("\n" + "="*70)
print("PART 4: DATA MANIPULATION")
print("="*70)

order_df['total_values'] = order_df['quantity'] * order_df['price_per_unit']
order_df['profit_margin'] = order_df['total_values'] * np.random.uniform(0.1, 0.5, len(order_df))
order_df['order_date_only'] = order_df['order_data'].dt.date

print(f"\nNew columns added:\n{order_df.to_string()}")
print('\n',order_df[['order_id', 'total_values', 'profit_margin', 'order_status']].head())

print(f"\nMissing values in order order_df:\n{order_df.isnull().sum()}")


print("\n" + "="*70)
print("PART 5: GROUPBY & AGGREGATION")
print("="*70)

products_state = order_df.groupby('product_name').agg({
    'order_id':'count',
    'quantity':'sum',
    'total_values':['sum', 'mean'],
    'profit_margin': 'sum'
}).round(2)

products_state.columns = ['Total_Orders', 'Total_Qty', 'Total_Revenue', 'Avg_Order_Value', 'Total_Profit']
print(f"\nproduct wise statistic")
print(products_state)


city_stats = order_df.merge(customers_df[['customer_id', 'city']], on='customer_id').groupby('city').agg({
    'total_values': 'sum',
    'order_id': 'count',
    'profit_margin': 'sum'
}).round(2)

city_stats.columns = ['Total_Revenue', 'Total_Orders', 'Total_Profit']
print("\n✓ City-wise Statistics:\n")
print(city_stats)
