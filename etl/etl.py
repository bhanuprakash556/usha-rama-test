# ============================================================
# SIMPLE NLP ETL PIPELINE
# Project: IMDb Movie Review Data
# ETL: Extract -> Transform -> Load
# ============================================================

# -------------------------
# 1. IMPORT LIBRARIES
# -------------------------

import pandas as pd
import re
import os
import urllib.request
import tarfile


# ============================================================
# 2. EXTRACT
# Download and extract IMDb Dataset
# ============================================================

print("STEP 1: EXTRACT")
print("-" * 50)

url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

zip_file = "./aclImdb.tar.gz"
dataset_folder = "aclImdb"


# Download dataset only if it doesn't exist
if not os.path.exists(dataset_folder):

    print("Downloading IMDb dataset...")

    urllib.request.urlretrieve(
        url,
        zip_file
    )

    print("Download completed!")

    print("Extracting dataset...")

    with tarfile.open(
        zip_file,
        "r:gz"
    ) as tar:

        tar.extractall("/content")

    print("Extraction completed!")

else:

    print("Dataset already exists!")


# ============================================================
# 3. READ REVIEWS
# ============================================================

print("\nReading movie reviews...")

data = []

# We will take 500 positive
# and 500 negative reviews

for sentiment in ["pos", "neg"]:

    folder = f"/content/aclImdb/train/{sentiment}"

    files = os.listdir(folder)

    # Take only first 500 reviews
    files = files[:500]

    for file_name in files:

        if file_name.endswith(".txt"):

            file_path = os.path.join(
                folder,
                file_name
            )

            # Read review
            with open(
                file_path,
                "r",
                encoding="utf-8"
            ) as file:

                review = file.read()

            # Store data
            data.append({

                "review": review,

                "sentiment": sentiment

            })


# Convert to DataFrame
df = pd.DataFrame(data)


print("Reviews extracted:", len(df))

print("\nRaw Data:")

print(df.head())


# ============================================================
# 4. TRANSFORM
# Clean the data
# ============================================================

print("\n\nSTEP 2: TRANSFORM")
print("-" * 50)


# Remove duplicate rows

df = df.drop_duplicates()


# Remove missing values

df = df.dropna(
    subset=[
        "review",
        "sentiment"
    ]
)


# ============================================================
# 5. CLEAN TEXT
# ============================================================

def clean_text(text):

    # Convert to lowercase
    text = text.lower()

    # Remove HTML tags
    text = re.sub(
        r"<.*?>",
        " ",
        text
    )

    # Remove URLs
    text = re.sub(
        r"http\S+|www\S+",
        " ",
        text
    )

    # Remove special characters
    text = re.sub(
        r"[^a-zA-Z\s]",
        " ",
        text
    )

    # Remove extra spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    # Remove spaces from beginning/end
    text = text.strip()

    return text


# Apply cleaning

df["clean_review"] = (

    df["review"]

    .apply(clean_text)

)


# ============================================================
# 6. FEATURE ENGINEERING
# Create simple NLP features
# ============================================================

# Number of characters

df["character_count"] = (

    df["clean_review"]

    .str.len()

)


# Number of words

df["word_count"] = (

    df["clean_review"]

    .str.split()

    .str.len()

)


# Number of exclamation marks

df["exclamation_count"] = (

    df["review"]

    .str.count("!")

)


# ============================================================
# 7. VIEW TRANSFORMED DATA
# ============================================================

print("\nTransformed Data:")

print(

    df[
        [
            "review",
            "clean_review",
            "sentiment",
            "word_count",
            "character_count"
        ]
    ].head()

)


# ============================================================
# 8. BEFORE VS AFTER CLEANING
# ============================================================

print("\n\nBEFORE CLEANING:")
print("-" * 50)

print(
    df.iloc[0]["review"][:500]
)


print("\n\nAFTER CLEANING:")
print("-" * 50)

print(
    df.iloc[0]["clean_review"][:500]
)


# ============================================================
# 9. DATA VALIDATION
# ============================================================

print("\n\nDATA VALIDATION")
print("-" * 50)

print(
    "Total records:",
    len(df)
)

print(
    "Missing values:"
)

print(
    df.isnull().sum()
)

print(
    "\nSentiment distribution:"
)

print(
    df["sentiment"].value_counts()
)


# ============================================================
# 10. LOAD
# Save processed data
# ============================================================

print("\n\nSTEP 3: LOAD")
print("-" * 50)


# Create output folder

os.makedirs(
    "/content/output",
    exist_ok=True
)


# Output file

output_file = (

    "/content/output/"
    "clean_movie_reviews.csv"

)


# Save CSV

df.to_csv(

    output_file,

    index=False

)


print(
    "Processed data saved successfully!"
)

print(
    "File location:",
    output_file
)


# ============================================================
# 11. FINAL RESULT
# ============================================================

print("\n\nETL PIPELINE COMPLETED!")
print("=" * 50)

print(
    "EXTRACT  -> IMDb Movie Reviews"
)

print(
    "TRANSFORM -> Clean Text + NLP Features"
)

print(
    "LOAD      -> clean_movie_reviews.csv"
)

print("=" * 50)


# Display final dataset

df.head()