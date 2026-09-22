import pandas as pd


data = {
    "name": [
        "John Doe",
        "  Alice Smith ",
        "Bob123",
        "Mary-Jane",
        "R@hul Kumar",
        "Tom_Johnson",
        "Ankit  Sharma",
        "Sara!",
        "john doe",
        "  PRIYA singh"
    ],
    "email": [
        "john@gmail.com",
        "alice.smith@yahoo.com",
        "bob123@gmail.com",
        "mary-jane@outlook.com",
        "rahul@gmail",
        "tom_johnson@company.org",
        "ankit@gmail.com",
        "sara!@gmail.com",
        "JOHN@GMAIL.COM",
        "priya.singh@gmail.com"
    ],
    "phone": [
        "9876543210",
        "+91-9876543210",
        "98765 43210",
        "(987) 654-3210",
        "98765432",
        "91 9876543210",
        "98765-43210",
        "12345",
        "+91 9876543210",
        "09876543210"
    ],
    "salary": [
        "₹50,000",
        "50000",
        "$60,000",
        "45,000 INR",
        "Rs. 70000",
        "₹80,000",
        "60000",
        "70k",
        "₹90,000",
        "1,00,000"
    ]
}

df = pd.DataFrame(data)

# print(df)

df["clean_name"] = (
    df["name"]
    .str.strip()
    .str.replace(r"[^A-Za-z\s]", "", regex=True)
    .str.replace(r"\s+", " ", regex=True)
    .str.title()
)

df["clean_name"].to_csv(
        'cleaned_data.csv',
        index=False
    )

df_from_csv = pd.read_csv("cleaned_data.csv")
print("\nData read from CSV:")
print(df_from_csv)

