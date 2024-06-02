from google.cloud import bigquery
from google.cloud import storage

def read_name_folders_bucket_gcs(bucket_name):
    """
    Read the name of folder inside the bucket of GCS
    Each folder represent a diffent dataset that will be used to explore the data, train models, etc
    """
    
    # create a client of storage
    client = storage.Client()
    
    
    # acess to bucket
    bucket = client.bucket(bucket_name)
    
    # list of objects in the bucket
    blobs = list(bucket.list_blobs())
    
    folders = set()
    
    for blob in blobs:
        # Separa el nombre del objeto por las barras diagonales
        parts = blob.name.split('/')
        if len(parts) > 1:
            # Si hay más de un elemento en la lista, es un folder
            folder_name = parts[0]
            folders.add(folder_name)

    # get a list of differents folders
    list_folders = list(folders)

    return list_folders