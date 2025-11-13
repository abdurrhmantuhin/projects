import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ===== PART 1: DATA CREATION (Simulating real data) =====
print("="*70)
print("E-COMMERCE SALES ANALYSIS PROJECT")
print("="*70)

# Create sample customer data
np.random.seed(42)
n_customers = 100
n_orders = 500

customers_data = {
    'customer_id': [f'CUST{i:04d}' for i in range(1, n_customers+1)],
    'customer_name': np.random.choice(['Ahmed', 'Fatima', 'Ali', 'Zainab', 'Hassan', 'Aisha', 'Omar', 'Leila', 'Karim', 'Noor'], n_customers),
    'email': [f'customer{i}@email.com' for i in range(1, n_customers+1)],
    'city': np.random.choice(['Dhaka', 'Chittagong', 'Sylhet', 'Khulna', 'Rajshahi'], n_customers),
    'signup_date': [datetime.now() - timedelta(days=np.random.randint(1, 365)) for _ in range(n_customers)]
}

customers_df = pd.DataFrame(customers_data)

# Create sales orders data
orders_data = {
    'order_id': [f'ORD{i:06d}' for i in range(1, n_orders+1)],
    'customer_id': np.random.choice(customers_df['customer_id'], n_orders),
    'product_name': np.random.choice(['Laptop', 'Phone', 'Tablet', 'Headphones', 'Monitor', 'Keyboard', 'Mouse', 'USB Cable'], n_orders),
    'quantity': np.random.randint(1, 5, n_orders),
    'price_per_unit': np.random.choice([500, 1000, 1500, 2000, 3000, 5000, 8000, 15000], n_orders),
    'order_date': [datetime.now() - timedelta(days=np.random.randint(1, 90)) for _ in range(n_orders)],
    'payment_method': np.random.choice(['Cash', 'Card', 'Mobile Banking', 'Bkash'], n_orders),
    'order_status': np.random.choice(['Completed', 'Pending', 'Cancelled'], n_orders, p=[0.7, 0.2, 0.1])
}

orders_df = pd.DataFrame(orders_data)

# Create product inventory data
products_data = {
    'product_name': ['Laptop', 'Phone', 'Tablet', 'Headphones', 'Monitor', 'Keyboard', 'Mouse', 'USB Cable'],
    'stock_quantity': [50, 200, 150, 300, 75, 400, 500, 1000],
    'category': ['Electronics', 'Electronics', 'Electronics', 'Accessories', 'Electronics', 'Accessories', 'Accessories', 'Accessories'],
    'supplier': ['Dell', 'Samsung', 'Apple', 'Sony', 'LG', 'Logitech', 'Razer', 'Generic']
}

products_df = pd.DataFrame(products_data)

print("\n✓ Data Created Successfully!")
print(f"\nCustomers: {len(customers_df)} | Orders: {len(orders_df)} | Products: {len(products_df)}")

# ===== PART 2: DATA EXPLORATION =====
print("\n" + "="*70)
print("PART 2: DATA EXPLORATION")
print("="*70)

print("\n--- Customers DataFrame Info ---")
print(customers_df.head())
print(f"\nShape: {customers_df.shape}")
print(f"Columns: {customers_df.columns.tolist()}")
print(f"\nData Types:\n{customers_df.dtypes}")

print("\n--- Orders DataFrame Info ---")
print(orders_df.head())
print(f"\nShape: {orders_df.shape}")

print("\n--- Products DataFrame Info ---")
print(products_df.head())
print(f"\nShape: {products_df.shape}")

# ===== PART 3: DATA FILTERING & SELECTION =====
print("\n" + "="*70)
print("PART 3: DATA FILTERING & SELECTION")
print("="*70)

# Filter 1: Orders over 15000 TK that are Completed
high_value_orders = orders_df[(orders_df['price_per_unit'] * orders_df['quantity'] > 15000) & 
                               (orders_df['order_status'] == 'Completed')]
print(f"\n✓ High Value Completed Orders (>15000 TK): {len(high_value_orders)}")
print(high_value_orders[['order_id', 'product_name', 'quantity', 'price_per_unit', 'order_status']].head())

# Filter 2: Customers from Dhaka
dhaka_customers = customers_df[customers_df['city'] == 'Dhaka']
print(f"\n✓ Dhaka Customers: {len(dhaka_customers)}")

# Filter 3: Electronic products that are low on stock
low_stock = products_df[(products_df['stock_quantity'] < 100) & (products_df['category'] == 'Electronics')]
print(f"\n✓ Low Stock Electronics: \n{low_stock}")

# ===== PART 4: DATA MANIPULATION =====
print("\n" + "="*70)
print("PART 4: DATA MANIPULATION")
print("="*70)

# Create new column for total order value
orders_df['total_value'] = orders_df['quantity'] * orders_df['price_per_unit']

# Create new column for order profit margin (simulated)
orders_df['profit_margin'] = orders_df['total_value'] * np.random.uniform(0.1, 0.4, len(orders_df))

# Create date-only column
orders_df['order_date_only'] = orders_df['order_date'].dt.date

print("\n✓ New columns added: total_value, profit_margin, order_date_only")
print(orders_df[['order_id', 'total_value', 'profit_margin', 'order_status']].head())

# Handle missing data (if any)
print(f"\nMissing values in orders_df:\n{orders_df.isnull().sum()}")

# ===== PART 5: GROUPBY & AGGREGATION =====
print("\n" + "="*70)
print("PART 5: GROUPBY & AGGREGATION")
print("="*70)

# Group by Product
product_stats = orders_df.groupby('product_name').agg({
    'order_id': 'count',
    'quantity': 'sum',
    'total_value': ['sum', 'mean'],
    'profit_margin': 'sum'
}).round(2)
product_stats.columns = ['Total_Orders', 'Total_Qty', 'Total_Revenue', 'Avg_Order_Value', 'Total_Profit']
print("\n✓ Product-wise Statistics:\n")
print(product_stats)

# Group by City
city_stats = orders_df.merge(customers_df[['customer_id', 'city']], on='customer_id').groupby('city').agg({
    'total_value': 'sum',
    'order_id': 'count',
    'profit_margin': 'sum'
}).round(2)
city_stats.columns = ['Total_Revenue', 'Total_Orders', 'Total_Profit']
print("\n✓ City-wise Statistics:\n")
print(city_stats)

# Group by Payment Method & Status
payment_status = orders_df.groupby(['payment_method', 'order_status']).agg({
    'order_id': 'count',
    'total_value': 'sum'
}).round(2)
payment_status.columns = ['Count', 'Revenue']
print("\n✓ Payment Method & Status:\n")
print(payment_status)

# ===== PART 6: SORTING =====
print("\n" + "="*70)
print("PART 6: SORTING")
print("="*70)

# Sort by highest revenue orders
top_orders = orders_df.nlargest(10, 'total_value')[['order_id', 'product_name', 'quantity', 'total_value', 'order_status']]
print("\n✓ Top 10 Orders by Revenue:\n")
print(top_orders)

# Sort by multiple columns
sorted_df = orders_df.sort_values(by=['order_status', 'total_value'], ascending=[True, False])
print("\n✓ Sorted by Status (ascending) then Revenue (descending) - First 5:\n")
print(sorted_df[['order_id', 'order_status', 'total_value']].head())

# ===== PART 7: MERGING & JOINING =====
print("\n" + "="*70)
print("PART 7: MERGING & JOINING")
print("="*70)

# Inner Merge: Orders with Customer Details
orders_with_customers = pd.merge(
    orders_df, 
    customers_df[['customer_id', 'customer_name', 'city', 'email']], 
    on='customer_id', 
    how='inner'
)
print(f"\n✓ Inner Merge (Orders + Customers): {len(orders_with_customers)} records")
print(orders_with_customers[['order_id', 'customer_name', 'city', 'product_name', 'total_value']].head())

# Left Merge: Orders with Product Details
orders_with_products = pd.merge(
    orders_df,
    products_df[['product_name', 'category', 'supplier']],
    on='product_name',
    how='left'
)
print(f"\n✓ Left Merge (Orders + Products): {len(orders_with_products)} records")
print(orders_with_products[['order_id', 'product_name', 'category', 'supplier']].head())

# Full Merge: All customer and order info combined
full_data = pd.merge(
    orders_with_customers,
    products_df[['product_name', 'category', 'stock_quantity']],
    on='product_name',
    how='left'
)
print(f"\n✓ Full Merge (Orders + Customers + Products): {len(full_data)} records")

# ===== PART 8: CONCATENATION =====
print("\n" + "="*70)
print("PART 8: CONCATENATION")
print("="*70)

# Split orders by status and concatenate
completed_orders = orders_df[orders_df['order_status'] == 'Completed'][['order_id', 'total_value']].head(5)
pending_orders = orders_df[orders_df['order_status'] == 'Pending'][['order_id', 'total_value']].head(5)

concat_vertical = pd.concat([completed_orders, pending_orders], axis=0, ignore_index=True)
print(f"\n✓ Vertical Concat (Completed + Pending Orders):\n")
print(concat_vertical)

# Concatenate horizontally
customers_sample = customers_df[['customer_id', 'customer_name']].head(5)
orders_sample = orders_df[['order_id', 'total_value']].head(5)
concat_horizontal = pd.concat([customers_sample, orders_sample], axis=1)
print(f"\n✓ Horizontal Concat (Customers + Orders):\n")
print(concat_horizontal)

# ===== PART 9: PIVOT TABLES =====
print("\n" + "="*70)
print("PART 9: PIVOT TABLES")
print("="*70)

# Create pivot table: Product vs Payment Method
pivot_1 = orders_df.pivot_table(
    values='total_value',
    index='product_name',
    columns='payment_method',
    aggfunc='sum',
    fill_value=0
).round(2)
print("\n✓ Pivot Table (Product vs Payment Method):\n")
print(pivot_1)

# Create pivot table: Product vs Status with count
pivot_2 = orders_df.pivot_table(
    values='order_id',
    index='product_name',
    columns='order_status',
    aggfunc='count',
    fill_value=0
)
print("\n✓ Pivot Table (Product vs Status - Order Count):\n")
print(pivot_2)

# ===== PART 10: EXPORTING DATA =====
print("\n" + "="*70)
print("PART 10: EXPORTING DATA")
print("="*70)

# Export different formats
full_data.to_csv('full_sales_data.csv', index=False)
full_data.to_excel('full_sales_data.xlsx', index=False)
full_data.to_json('full_sales_data.json', index=False)

product_stats.to_csv('product_statistics.csv')
city_stats.to_csv('city_statistics.csv')
pivot_1.to_csv('pivot_payment_method.csv')

print("\n✓ Files Exported:")
print("  - full_sales_data.csv")
print("  - full_sales_data.xlsx")
print("  - full_sales_data.json")
print("  - product_statistics.csv")
print("  - city_statistics.csv")
print("  - pivot_payment_method.csv")

# ===== PART 11: ADVANCED ANALYSIS =====
print("\n" + "="*70)
print("PART 11: ADVANCED ANALYSIS")
print("="*70)

# Calculate customer lifetime value
customer_ltv = orders_df.merge(customers_df[['customer_id', 'city']], on='customer_id').groupby('customer_id').agg({
    'total_value': 'sum',
    'order_id': 'count',
    'city': 'first'
}).round(2)
customer_ltv.columns = ['Total_Spent', 'Order_Count', 'City']
customer_ltv = customer_ltv.sort_values('Total_Spent', ascending=False)
print("\n✓ Top 10 Customers by Lifetime Value:\n")
print(customer_ltv.head(10))

# Performance by Category
category_performance = orders_df.merge(
    products_df[['product_name', 'category']], 
    on='product_name'
).groupby('category').agg({
    'total_value': 'sum',
    'profit_margin': 'sum',
    'order_id': 'count'
}).round(2)
category_performance.columns = ['Revenue', 'Profit', 'Orders']
category_performance['Profit_Margin_%'] = (category_performance['Profit'] / category_performance['Revenue'] * 100).round(2)
print("\n✓ Category Performance:\n")
print(category_performance)

# Monthly sales trend
orders_df['month'] = orders_df['order_date'].dt.to_period('M')
monthly_sales = orders_df.groupby('month').agg({
    'total_value': 'sum',
    'order_id': 'count',
    'profit_margin': 'sum'
}).round(2)
monthly_sales.columns = ['Revenue', 'Orders', 'Profit']
print("\n✓ Monthly Sales Trend:\n")
print(monthly_sales)

print("\n" + "="*70)
print("PROJECT COMPLETED SUCCESSFULLY!")
print("="*70)