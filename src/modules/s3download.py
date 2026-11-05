import hashlib
import logging
import os
import re
import urllib.parse

import boto3
import botocore

logger = logging.getLogger(__name__)


def download_file_from_s3(s3_key):
    if s3_key:
        if urllib.parse.urlparse(s3_key).scheme:
            logger.info(f"S3 key is a URL: {s3_key}")
            bucket_name = os.getenv('BUCKET_NAME')
            pattern = f"{bucket_name}/([^']+)"
            match = re.search(pattern, s3_key)
            if match:
                s3_key = match.group(1)
                logger.info(f"Extracted S3 key: {s3_key}")
                s3_key_bytes = s3_key.encode('utf-8')
                Id = hashlib.sha256(s3_key_bytes).hexdigest()
                print("url Id", Id)

            else:
                logger.error(f"Failed to extract S3 key from URL: {s3_key}")
                return None
        else:
            logger.info(f"S3 key is : {s3_key}")
            s3_key_bytes = s3_key.encode('utf-8')
            Id = hashlib.sha256(s3_key_bytes).hexdigest()

    logger.info(f"Downloading file from {s3_key}")

    temp_folder_path = '/tmp'

    # Extract the filename from the S3 key
    file_name = os.path.basename(s3_key)

    # Construct the full file path within the 'temp' directory
    file_path = os.path.join(temp_folder_path, file_name)

    try:
        # Initialize a session using AWS credentials and region
        session = boto3.Session()
        AccessKey = os.getenv('ACCESS_KEY')
        Secret = os.getenv('SECRET_KEY')
        # Create an S3 client
        s3_client = boto3.client('s3', aws_access_key_id=AccessKey, aws_secret_access_key=Secret)
        bucket_name = os.getenv('BUCKET_NAME')
        # Download the file from S3 bucket to a local file
        s3_client.download_file(bucket_name, s3_key, file_path)

        logger.info(f"File downloaded successfully to: {file_path}")
        return file_path
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            logger.info(f"The object '{s3_key}' does not exist in bucket '{bucket_name}'")
        else:
            logger.info(f"Error downloading file from S3: {e}")
