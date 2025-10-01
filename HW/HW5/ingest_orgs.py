import sys
import pysqlite3
sys.modules["sqlite3"] = pysqlite3
import os
import zipfile
import chromadb
from bs4 import BeautifulSoup

# --------------------------
# 1. Paths
# --------------------------
zip_path = "HW/HW5/su_orgs.zip"
extract_path = "HW/HW5/su_orgs_data"

# Extract if not already extracted
if not os.path.exists(extract_path):
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_path)
print(f"✅ Extracted {zip_path} to {extract_path}")

# --------------------------
# 2. Recursively read all HTML files
# --------------------------
documents = []
ids = []

for root, _, files in os.walk(extract_path):  # ✅ Walk all subfolders
    for idx, filename in enumerate(files):
        if filename.endswith(".html"):
            filepath = os.path.join(root, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                html = f.read()
                soup = BeautifulSoup(html, "html.parser")
                text = soup.get_text(separator=" ", strip=True)
                if text:
                    documents.append(text)
                    ids.append(f"doc_{idx}")

print(f"📄 Loaded {len(documents)} documents from {extract_path}")

# --------------------------
# 3. Store in ChromaDB
# --------------------------
chroma_client = chromadb.PersistentClient(path="clubs_db")
collection = chroma_client.get_or_create_collection(name="su_orgs")

# Clear duplicates
if collection.count() > 0:
    existing_ids = collection.get()["ids"]
    collection.delete(ids=existing_ids)

if documents:
    collection.add(
        documents=documents,
        ids=ids
    )

print(f"✅ Inserted {len(documents)} documents into ChromaDB (su_orgs collection).")
