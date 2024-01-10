import runpod
import time
import logging
import traceback
from sys import exc_info
from transcriber import transcriber_v2_nemo_asr

# Configure the logging system
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def transcribe_text_from_url(urls):
    try:
        st = time.time()
        logging.info(f"Passing URLs to transcriber...")
        text = transcriber_v2_nemo_asr(urls)
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
    urls = event['input']['audio_urls']
    logging.info("Received event with audio URLs")

    text = transcribe_text_from_url(urls)
    return {"output": text}


runpod.serverless.start({
    "handler": handler
})
