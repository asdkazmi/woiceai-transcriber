import os

import runpod
import time
import logging
import traceback
from sys import exc_info
from s3download import download_file_from_s3
from transcriber import transcriber_v2_nemo_asr

# Configure the logging system
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def transcribe_text_from_url(urls, from_file=False):
    try:
        st = time.time()
        logging.info(f"Passing URLs to transcriber...")
        text = transcriber_v2_nemo_asr(urls=urls, from_file=from_file)
        et = time.time()
        t = et - st
        logging.info(f"Total inference time {t:.3f} seconds ")
        return text

    except AssertionError:
        logger.error(traceback.format_exc())
        return {
            'statusCode': 400,
            'body': {
                'error': f"{exc_info()[1]}"
            }
        }

    except:
        logger.error(traceback.format_exc())
        return {
            'statusCode': 500,
            'body': {
                'error': f"{str(exc_info()[0]).split('class')[1].split('>')[0].strip()}: {exc_info()[1]}"
            }
        }

def handler(event):
    inputs = event['input']
    urls = inputs.get('audio_urls')
    s3_key = inputs.get('s3_key')
    if urls:
        logging.info("Received event with audio URLs")
        text = transcribe_text_from_url(urls)
        return {"output": text}
    elif s3_key:
        logging.info("Received event with s3_key")
        file_path = download_file_from_s3(s3_key)
        if file_path:
            text = transcribe_text_from_url(urls=[file_path], from_file=True)
            try:
                os.remove(file_path)
                logging.info(f"Successfully removed file: {file_path}")
            except Exception as e:
                logging.error(f"Error removing file: {e}")
            return {"output": text}
        else:
            return {"output": 'Failed to download audio file'}

runpod.serverless.start({
    "handler": handler
})
