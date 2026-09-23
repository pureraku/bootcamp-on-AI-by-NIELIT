import pandas as pd
import numpy as np

# ============================================
# 1. SETTINGS
# ============================================

np.random.seed(42)

N = 1_000_000

# ============================================
# 2. MASTER DATA
# ============================================

first_names = np.array([
    "Rahul", "Amit", "Priya", "Neha", "Ankit",
    "Rohit", "Sneha", "Vikas", "Pooja", "Arjun",
    "Karan", "Simran", "Aditya", "Riya", "Nikhil",
    "Meera", "Varun", "Isha", "Manish", "Kavya"
])

last_names = np.array([
    "Sharma", "Kumar", "Singh", "Patel", "Gupta",
    "Verma", "Mehta", "Joshi", "Malhotra", "Kapoor",
    "Reddy", "Mishra", "Chopra", "Agarwal", "Sinha"
])

cities = np.array([
    "Delhi", "Mumbai", "Bangalore", "Hyderabad",
    "Chennai", "Kolkata", "Pune", "Ahmedabad",
    "Jaipur", "Lucknow", "Chandigarh", "Indore",
    "Surat", "Nagpur", "Bhopal"
])

states = np.array([
    "Delhi", "Maharashtra", "Karnataka", "Telangana",
    "Tamil Nadu", "West Bengal", "Maharashtra",
    "Gujarat", "Rajasthan", "Uttar Pradesh",
    "Chandigarh", "Madhya Pradesh", "Gujarat",
    "Maharashtra", "Madhya Pradesh"
])

products = np.array([
    "Laptop",
    "Smartphone",
    "Tablet",
    "Monitor",
    "Keyboard",
    "Mouse",
    "Headphones",
    "Smartwatch",
    "Camera",
    "Printer",
    "Power Bank",
    "USB Cable",
    "Gaming Chair",
    "Desk",
    "Webcam"
])

categories = np.array([
    "Electronics",
    "Electronics",
    "Electronics",
    "Electronics",
    "Accessories",
    "Accessories",
    "Accessories",
    "Electronics",
    "Electronics",
    "Electronics",
    "Accessories",
    "Accessories",
    "Furniture",
    "Furniture",
    "Accessories"
])

payment_methods = np.array([
    "UPI",
    "Credit Card",
    "Debit Card",
    "Net Banking",
    "Cash on Delivery",
    "Wallet"
])

order_statuses = np.array([
    "Delivered",
    "Shipped",
    "Processing",
    "Cancelled",
    "Returned"
])

shipping_types = np.array([
    "Standard",
    "Express",
    "Same Day"
])

genders = np.array([
    "Male",
    "Female",
    "Other"
])

# ============================================
# 3. GENERATE BASIC DATA
# ============================================

customer_id = np.random.randint(
    100000,
    250000,
    N
)

first = np.random.choice(
    first_names,
    N
)

last = np.random.choice(
    last_names,
    N
)

customer_name = np.char.add(
    np.char.add(first, " "),
    last
)

# Email
email = np.char.lower(
    np.char.add(
        np.char.add(first, "."),
        np.char.add(
            last,
            "@gmail.com"
        )
    )
)

# ============================================
# 4. ORDER DATA
# ============================================

order_id = np.arange(
    10000000,
    10000000 + N
)

product_index = np.random.randint(
    0,
    len(products),
    N
)

product = products[product_index]

category = categories[product_index]

city_index = np.random.randint(
    0,
    len(cities),
    N
)

city = cities[city_index]

state = states[city_index]

gender = np.random.choice(
    genders,
    N,
    p=[0.48, 0.49, 0.03]
)

payment = np.random.choice(
    payment_methods,
    N,
    p=[0.35, 0.25, 0.18, 0.08, 0.10, 0.04]
)


status = np.random.choice(
    order_statuses,
    N,
    p=[
        0.70,  # Delivered
        0.10,  # Shipped
        0.05,  # Processing
        0.05,  # Cancelled
        0.10   # Returned
    ]
)

shipping = np.random.choice(
    shipping_types,
    N,
    p=[0.70, 0.25, 0.05]
)

# ============================================
# 5. DATES
# ============================================

start_date = np.datetime64("2023-01-01")
end_date = np.datetime64("2025-12-31")

days = (
    end_date - start_date
).astype(int)

random_days = np.random.randint(
    0,
    days,
    N
)

order_date = (
    start_date +
    random_days.astype("timedelta64[D]")
)

# ============================================
# 6. FINANCIAL DATA
# ============================================

# Base price for each product
product_prices = {
    "Laptop": 65000,
    "Smartphone": 30000,
    "Tablet": 22000,
    "Monitor": 18000,
    "Keyboard": 2500,
    "Mouse": 1200,
    "Headphones": 4500,
    "Smartwatch": 7000,
    "Camera": 45000,
    "Printer": 12000,
    "Power Bank": 1800,
    "USB Cable": 500,
    "Gaming Chair": 15000,
    "Desk": 12000,
    "Webcam": 3500
}

base_price = np.array([
    product_prices[p]
    for p in product
])

# Random price variation
unit_price = (
    base_price *
    np.random.uniform(
        0.85,
        1.15,
        N
    )
)

unit_price = np.round(
    unit_price,
    2
)

quantity = np.random.choice(
    [1, 2, 3, 4, 5],
    N,
    p=[0.55, 0.25, 0.12, 0.06, 0.02]
)

discount = np.random.choice(
    [0, 5, 10, 15, 20, 25, 30],
    N,
    p=[0.30, 0.20, 0.20, 0.12, 0.10, 0.05, 0.03]
)

gross_amount = (
    unit_price *
    quantity
)

discount_amount = (
    gross_amount *
    discount / 100
)

net_amount = (
    gross_amount -
    discount_amount
)

# Tax
tax_rate = np.random.choice(
    [0.05, 0.12, 0.18],
    N,
    p=[0.20, 0.30, 0.50]
)

tax_amount = (
    net_amount *
    tax_rate
)

total_amount = (
    net_amount +
    tax_amount
)

# ============================================
# 7. CUSTOMER ATTRIBUTES
# ============================================

age = np.random.randint(
    18,
    70,
    N
)

customer_rating = np.round(
    np.random.uniform(
        1,
        5,
        N
    ),
    1
)

# ============================================
# 8. CREATE DATAFRAME
# ============================================

df = pd.DataFrame({
    "order_id": order_id,
    "customer_id": customer_id,
    "customer_name": customer_name,
    "email": email,
    "gender": gender,
    "age": age,
    "city": city,
    "state": state,
    "order_date": order_date,
    "product": product,
    "category": category,
    "quantity": quantity,
    "unit_price": unit_price,
    "discount_percent": discount,
    "gross_amount": np.round(gross_amount, 2),
    "discount_amount": np.round(discount_amount, 2),
    "tax_rate": tax_rate,
    "tax_amount": np.round(tax_amount, 2),
    "total_amount": np.round(total_amount, 2),
    "payment_method": payment,
    "order_status": status,
    "shipping_type": shipping,
    "customer_rating": customer_rating
})

# ============================================
# 9. ADD REALISTIC DATA PROBLEMS
# ============================================

# Missing emails
missing_email = np.random.random(N) < 0.01

df.loc[
    missing_email,
    "email"
] = np.nan

# Missing ages
missing_age = np.random.random(N) < 0.005

df.loc[
    missing_age,
    "age"
] = np.nan

# Messy customer names
messy_names = np.random.random(N) < 0.01

df.loc[
    messy_names,
    "customer_name"
] = (
    "  " +
    df.loc[messy_names, "customer_name"] +
    " "
)

# Random uppercase emails
uppercase_email = np.random.random(N) < 0.01

df.loc[
    uppercase_email,
    "email"
] = (
    df.loc[uppercase_email, "email"]
    .str.upper()
)

# ============================================
# 10. SAVE TO CSV
# ============================================

df.to_csv(
    "ecommerce_transactions.csv",
    index=False
)

print("Dataset created successfully!")
print("Rows:", len(df))
print("Columns:", len(df.columns))
print("File: ecommerce_transactions.csv")

print("\nFirst 5 rows:")
print(df.head())
