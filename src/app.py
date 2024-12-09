import os
import json
import runpod
import time
import logging
import traceback

from sys import exc_info
from modules import chunks
from modules.credits import cost
from modules.embedings import runpod_api_pipeline
from modules.mongo import insert_audio
from modules.s3download import download_file_from_s3
from modules.converting_to_wav import convert_audio_to_wav
from modules.sns import sending_message_attributes_to_sns
from transcriber import transcriber_v2_nemo_asr, transcribe_without_diarization

# Configure the logging system
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def transcribe_text_from_url(urls, from_file=False, diarization=False):
    try:
        st = time.time()
        logging.info(f"Passing URLs to transcriber...")
        if diarization:
            text = transcriber_v2_nemo_asr(urls=urls, from_file=from_file)
        else:
            text = transcribe_without_diarization(paths=urls, from_file=from_file)
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
    handler_start = time.time()
    inputs = event['input']
    urls = inputs.get('audio_urls')
    s3_key = inputs.get('s3_key')
    kbId = inputs.get('kbId')
    content_type = inputs.get('content_type')
    title = inputs.get('title')
    diarization = inputs.get('diarization', False)
    text = ""

    if urls:
        logging.info("Received event with audio URLs")
        text = transcribe_text_from_url(urls, diarization=diarization)
    elif s3_key:
        logging.info("Received event with s3_key")
        file_path = download_file_from_s3(s3_key)
        if not file_path.lower().endswith('.wav'):
            file_path = convert_audio_to_wav(file_path)
        text = transcribe_text_from_url(urls=[file_path], from_file=True, diarization=diarization)
        os.remove(file_path)
    try:
        transcript = text.get("transcription")
        logger.info("Audio file transcribed successfully")

        logger.info("Converting transcript to chunks")
        audio_chunks = chunks.process_transcript_and_create_chunks(content=transcript)
        combined_transcript = " ".join(audio_chunks)
        prefix = f"File Type: Audio; Audio Name: {title}: \n\n"
        updated_audio_chunks = [prefix + item for item in audio_chunks]

        logger.info("Creating embeddings for chunks")
        embeddings, execution_time_embeddings = runpod_api_pipeline(substrings=updated_audio_chunks, query='',
                                                                    top_k=5,
                                                                    threshold=100,
                                                                    task='embeddings', embedings=True)

        logger.info("Uploading to DB")
        insert_audio(audio_chunks, embeddings, kbId, s3_key, title)
        handler_end = time.time()
        handler_duration = handler_end - handler_start
        credits_consumed = cost(handler_duration, execution_time_embeddings)
        message_attributes = {
            "KNOWLEDGEBASE.ID": {"DataType": "String", "StringValue": kbId},
            "KNOWLEDGEBASE.CONTENT": {"DataType": "String",
                                      "StringValue": s3_key if s3_key else urls},
            "KNOWLEDGEBASE.CONTENT_TYPE": {"DataType": "String", "StringValue": content_type},
            "KNOWLEDGEBASE.STATUS": {"DataType": "String", "StringValue": "Completed"},
            "KNOWLEDGEBASE.CREDITS_CONSUMED": {"DataType": "String", "StringValue": str(credits_consumed)},
            "KNOWLEDGEBASE.TRANSCRIPTION": {"DataType": "String", "StringValue": combined_transcript}
        }
        sending_message_attributes_to_sns(message_attributes)
        return {"KNOWLEDGEBASE.ID": {"DataType": "String", "Value": kbId},
                "KNOWLEDGEBASE.CONTENT": {"DataType": "String",
                                          "Value": s3_key if s3_key else urls,
                                          "transcription": combined_transcript},
                "KNOWLEDGEBASE.CONTENT_TYPE": {"DataType": "String", "Value": content_type},
                "KNOWLEDGEBASE.STATUS": {"DataType": "String", "Value": "Completed"},
                "KNOWLEDGEBASE.CREDITS_CONSUMED": {"DataType": "String", "Value": str(credits_consumed)}}



    except Exception as e:
        logger.error(f"Error transcribing audio file: {e}")
        message_attributes = {
            "KNOWLEDGEBASE.ID": {"DataType": "String", "StringValue": kbId},
            "KNOWLEDGEBASE.CONTENT": {"DataType": "String", "StringValue": s3_key if s3_key else urls},
            "KNOWLEDGEBASE.CONTENT_TYPE": {"DataType": "String", "StringValue": content_type},
            "KNOWLEDGEBASE.STATUS": {"DataType": "String", "StringValue": "Failed"},
            "KNOWLEDGEBASE.CREDITS_CONSUMED": {"DataType": "String", "StringValue": "0"}
        }
        sending_message_attributes_to_sns(message_attributes)
        return {"KNOWLEDGEBASE.ID": {"DataType": "String", "Value": kbId},
                "KNOWLEDGEBASE.CONTENT": {"DataType": "String",
                                          "Value": s3_key if s3_key else urls},
                "KNOWLEDGEBASE.CONTENT_TYPE": {"DataType": "String", "Value": content_type},
                "KNOWLEDGEBASE.STATUS": {"DataType": "String", "Value": "Failed"},
                "KNOWLEDGEBASE.CREDITS_CONSUMED": {"DataType": "String", "Value": "0"}}


runpod.serverless.start({
    "handler": handler
})
