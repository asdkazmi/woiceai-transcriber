import logging
import requests
import time
from dotenv import load_dotenv
import os

load_dotenv()

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def make_request(base_url, endpoint, payload=None, method="post"):
    """
    The make_request function sends a request to the RUNPOD API with a specified method (POST or GET) and returns the
    response.
    :param base_url:
    :param endpoint:
    :param payload:
    :param method:
    :return:
    """
    url = f"{base_url}/{endpoint}"
    runpod_api_key = os.getenv('RUNPOD_API_KEY')
    headers = {"Authorization": f"Bearer {runpod_api_key}"}
    if method == "post":
        response = requests.post(url, headers=headers, json=payload)
        return response.status_code, response.json()
    elif method == "get":
        response = requests.get(url, headers=headers)
        return response.status_code, response.json()


def wait_for_completion(base_url, queue_id, threshold):
    """
    The wait_for_completion function waits for a request to complete and returns the result.
    :param base_url:
    :param queue_id:
    :param threshold:
    :return:
    """
    start_time = time.time()

    while time.time() - start_time < threshold:
        status_code, response = make_request(base_url, f"status/{queue_id}", method="get")

        if status_code == 200 and response.get('status') == 'COMPLETED':
            return response

        time.sleep(2)

    return None


def cancel_request(base_url, queue_id):
    """
    The cancel_request function cancels a request and returns a status message.
    :param base_url:
    :param queue_id:
    :return:
    """
    logger.info("Threshold reached, cancelling request")

    cancel_code, cancel_response = make_request(base_url, f"cancel/{queue_id}", method="post")

    if cancel_code == 200:
        return {"status": "Request cancelled"}
    else:
        return {"status": "Unable to cancel request"}


def runpod_api_pipeline(substrings=None, audio_urls=None, audio_s3_key=None, query='', top_k=5, threshold=100, task='embeddings',
                        embedings=False):
    """

    :param substrings:
    :param audio_urls:
    :param query:
    :param top_k:
    :param threshold:
    :param task:
    :param embedings:
    :return:
    """
    if embedings:
        base_url = os.getenv('RUNPOD_BASE_URL')
        data_payload = {
            "input": {"text": substrings, "task": task}} if substrings is not None else {}

    method_a_response_code, method_a_response = make_request(base_url=base_url, endpoint="runsync",
                                                             payload=data_payload)

    if method_a_response_code == 200 and method_a_response.get("status") == 'COMPLETED':
        response = method_a_response["output"]["output"]
        execution_time = method_a_response['executionTime'] / 1000  # conv into seconds
        return response, execution_time

    elif method_a_response.get("status") == "IN_PROGRESS" or method_a_response.get("status") == "IN_QUEUE":
        in_progress_id = method_a_response.get('id')
        # logger.info("Q. ID: ", in_progress_id)
        result = wait_for_completion(base_url, in_progress_id, threshold)

        if result:
            return result["output"]["output"]

        return cancel_request(base_url, in_progress_id)

    else:
        logger.info("RUNSYNC API failed, attempting RUN API")
        method_b_response_code, method_b_response = make_request(base_url=base_url, endpoint="run",
                                                                 payload=data_payload)

        if method_b_response_code == 200:
            queue_id = method_b_response.get('id')
            result = wait_for_completion(base_url, queue_id, threshold)

            if result:
                return result["output"]["output"]

            return cancel_request(base_url, queue_id)

        else:
            logger.info("Both methods failed")
            return {"status": "Both methods failed"}
