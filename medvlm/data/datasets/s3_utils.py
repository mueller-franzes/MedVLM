from io import BytesIO 
import boto3
from boto3.s3.transfer import TransferConfig
from nibabel import FileHolder, Nifti1Image
from gzip import GzipFile
import torchio as tio 
import json

def init_bucket(profile_name=None, endpoint_url=None, bucket_name=None, config_file="s3_config.json"):
    """
    Initialize an S3 bucket connection using parameters loaded from a configuration file.

    Parameters:
        profile_name (str): The AWS profile name to use. Overrides config file if provided.
        endpoint_url (str): The S3 endpoint URL. Overrides config file if provided.
        bucket_name (str): The S3 bucket name. Overrides config file if provided.
        config_file (str): Path to the JSON configuration file containing the parameters.

    Returns:
        s3.Bucket: A boto3 S3 bucket resource.
    """
    # Load parameters from the configuration file if not explicitly provided
    try:
        with open(config_file, "r") as file:
            config = json.load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file '{config_file}' not found.")

    # Use values from the config file only if they are not provided as arguments
    profile_name = profile_name or config.get("profile_name")
    endpoint_url = endpoint_url or config.get("endpoint_url")
    bucket_name = bucket_name or config.get("bucket_name")

    if not all([profile_name, endpoint_url, bucket_name]):
        raise ValueError("Missing required parameters. Ensure all parameters are provided or exist in the config file.")

    # Create a session using the specified profile
    session = boto3.Session(profile_name=profile_name)

    # Use the session to create an S3 resource with the specified endpoint
    s3 = session.resource('s3', endpoint_url=endpoint_url)

    # Get the specific bucket
    return s3.Bucket(bucket_name)




def load_bytesio(bucket, path_file):
    # # Retrieve the specific object
    # response = bucket.Object(str(path_file)).get()

    # # Read the content of the object (file)
    # file_content = response['Body'].read()

    # # Create an in-memory binary stream
    # file_stream = BytesIO(file_content)

    config = TransferConfig(
        multipart_threshold=10 * 1024 * 1024,  
        max_concurrency=10, 
        max_io_queue=100,
        multipart_chunksize=10 * 1024 * 1024,  
        io_chunksize=1* 1024 * 1024,
    )
    file_stream = BytesIO()
    bucket.download_fileobj(str(path_file), file_stream, Config=config)
    file_stream.seek(0) 

    return file_stream



def byteio2nibabel(file_stream, gzip=True):
    fileobj = GzipFile(fileobj=file_stream) if gzip else file_stream
    fh = FileHolder(fileobj=fileobj)
    return Nifti1Image.from_file_map({'header': fh, 'image': fh})


def byteio2torchio(file_stream, img_type=tio.IMAGE, gzip=True):
    img = byteio2nibabel(file_stream, gzip)
    if img_type == tio.IMAGE:
        return tio.ScalarImage(tensor=img.get_fdata()[None], affine=img.affine)
    elif img_type == tio.LABEL:
        return tio.LabelMap(tensor=img.get_fdata()[None], affine=img.affine)
    else:
        raise ValueError(f"Unknown type: {img_type}")
    

def load_torchio(bucket, path_file, img_type=tio.IMAGE, gzip=True):
    fstream = load_bytesio(bucket, path_file)
    return byteio2torchio(fstream, img_type, gzip)