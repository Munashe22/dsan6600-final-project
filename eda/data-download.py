from google.cloud import storage
import os

client = storage.Client.create_anonymous_client()

bucket_name = "brb-traffic"
bucket = client.bucket(bucket_name)

local_folder = "data"

os.makedirs(local_folder, exist_ok=True)

blobs = bucket.list_blobs(prefix="added/")

for blob in blobs:
    local_path = os.path.join(local_folder, blob.name)

    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    print(blob.name)
    blob.download_to_filename(local_path)