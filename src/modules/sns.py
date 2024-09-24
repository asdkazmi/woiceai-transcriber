import json
import os
import boto3
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


def sending_message_attributes_to_sns(message_attributes):

    access_key = os.environ.get("ACCESS_KEY")
    secret_key = os.environ.get("SECRET_KEY")
    sns_client = boto3.client('sns', region_name='us-east-1', aws_access_key_id=access_key,
                              aws_secret_access_key=secret_key)

    try:
        topic_arn = os.environ.get("TOPIC_ARN")
        message = {
            "text": "This is a message from lambda to sns."
        }
        response = sns_client.publish(
            TopicArn=topic_arn,
            Message=json.dumps(message),
            MessageAttributes=message_attributes
        )
        logger.info(f"Message ID: {response['MessageId']}")

    except Exception as e:
        logger.error(f"Error publishing message to SNS: {e}")
        raise
