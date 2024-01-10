import os
import time
import librosa
import numpy as np
import requests
from io import BytesIO
import logging
from models import asr_model, pipe
import soundfile as sf
import base64

# Configure the logging system
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def normalize_timeline(timeline):
    length = len(timeline)
    prev_start = 0
    prev_end = 0

    timeline_norm = list()

    for i, (start, end, speaker) in enumerate(timeline):

        index = min(i + 1, length - 1)

        next_start = timeline[index][0]
        next_end = timeline[index][1]

        if i == 0:
            new_start = timeline[0][0] / 2
            new_end = (end + next_start) / 2

        else:
            diff = start - prev_end
            if (diff > 22050):
                new_start = start - 10000

                if end < next_start:
                    diff = next_start - end
                    if diff > 22050:
                        new_end = end + 10000
                    else:
                        new_end = (end + next_start) / 2
                else:
                    new_end = end

            else:
                if prev_end < start:
                    new_start = (prev_end + start) / 2
                else:
                    new_start = start

                if end < next_start:
                    new_end = (end + next_start) / 2
                else:
                    new_end = end

        if start > prev_start and end < prev_end:
            # if start < end:
            new_start = start
            new_end = end
        # else:
        # print(f'i:{i}, start: {start}, end: {end}')

        timeline_norm.append([new_start, new_end, speaker])

        prev_start = start
        prev_end = end

    return timeline_norm


def normalize_timeline_v2(timeline):
    length = len(timeline)
    prev_start = 0
    prev_end = 0
    prev_speaker = ""

    timeline_norm = list()

    for i, (start, end, speaker) in enumerate(timeline):

        if i == 0:
            prev_start = start
            prev_end = end
            prev_speaker = speaker

        if speaker != prev_speaker:
            new_start = prev_start
            new_end = prev_end

            timeline_norm.append([new_start, new_end, prev_speaker])

            prev_start = start
            prev_speaker = speaker

        if i == length - 1:
            timeline_norm.append([prev_start, end, speaker])

        prev_end = end

    # return normalize_timeline(timeline_norm)
    return timeline_norm


def retrieve_data_from_urls(urls):
    all_audios = []
    diarization_list = []
    st = time.time()

    # Filter out empty strings from urls
    urls = [url for url in urls if url]

    for url in urls:
        response = requests.get(url)
        response.raise_for_status()

        audio = BytesIO(response.content)

        d = pipe(audio, min_speakers=1, max_speakers=10)
        diarization_list.append(d)

        all_audios.append(librosa.load(audio))

    et = time.time()
    t = et - st
    logging.info(f'Diarization time: {t:.3f} sec')
    return all_audios, diarization_list


def retrieve_data_from_file(urls):
    all_audios = []
    diarization_list = []
    st = time.time()

    # Filter out empty strings from urls
    urls = [url for url in urls if url]

    logging.info("Audio diarization in progress...")

    for url in urls:
        d = pipe(url, min_speakers=1, max_speakers=10)
        diarization_list.append(d)

        all_audios.append(librosa.load(url))

    et = time.time()
    t = et - st
    logging.info(f'Diarization time: {t:.3f} sec')
    return all_audios, diarization_list


def process_diarizations(diarizations):
    timelines = []
    for diarization in diarizations:
        timeline = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            timeline.append([turn.start, turn.end, speaker])
        timelines.append(timeline)

    return timelines


def transcriber_v2_nemo_asr(urls, from_file=False):
    stt = time.time()

    # Filter out empty strings from urls
    if isinstance(urls, list):
        urls = [url for url in urls if url]
    else:
        logging.info("The provided variable is not a list.")
        return None

    if from_file:
        audios, diarizations = retrieve_data_from_file(urls)
    else:
        audios, diarizations = retrieve_data_from_urls(urls)

    timelines = process_diarizations(diarizations)

    num_of_urls = len(urls)
    transcriptions = []

    for i in range(num_of_urls):
        logging.info(f"Processing audio URL: {i + 1}/{num_of_urls}")

        # Extract unique speakers
        unique_speakers = set(entry[2] for entry in timelines[i])

        # Convert the set to a list if needed
        unique_speakers_list = list(unique_speakers)

        # Print the unique speakers
        logging.info(f'No. of speakers: {unique_speakers_list}')

        if len(unique_speakers_list) == 1:
            audio, sr = audios[i]
            output_path = "output.wav"
            sf.write(output_path, audio, sr)
            output = asr_model.transcribe(paths2audio_files=[output_path], batch_size=4)
            transcription = output[0][0]

            transcriptions.append([{
                "text": transcription,
                "timestamp": {
                    "start": timelines[i][0][0],
                    "end": timelines[i][0][1]
                },
                # "name": timelines[i][0][2],
                "speaker": int(timelines[i][0][2].lower().replace("speaker_", "").strip())
                # "audio": ""
                # "audio": base64.b64encode(audios[i][0]).decode('utf-8')
            }])

        else:
            timeline_norm = normalize_timeline_v2(timelines[i])

            transcription, audio_chunks = transcribe_audio_with_classification_v2_nemo_asr(audios[i], timeline_norm)

            transcription_list = list()

            for j, text in enumerate(transcription):
                transcription_list.append({
                    "text": text['text'],
                    "timestamp": {
                        "start": timeline_norm[j][0],
                        "end": timeline_norm[j][1]
                    },
                    # "name": timeline_norm[j][2],
                    "speaker": int(timeline_norm[j][2].lower().replace("speaker_", "").strip())
                    # "audio": ""
                    # "audio": base64.b64encode(audio_chunks[j]).decode('utf-8')

                })

            transcriptions.append(transcription_list)

    ett = time.time()
    t = ett - stt
    logging.info(f'Inference time: {t:.3f} sec')

    return {"status": "completed",
            "transcription": transcriptions
            }


def transcribe_audio_with_classification_v2_nemo_asr(audio, timeline_norm):
    logging.info("Audio to text transcription with classification")

    audio1, sr = audio

    length = len(audio1)
    dur = length / sr
    factor = sr  # length/dur
    logging.info(f"Audio duration: {dur:.3f} sec")

    threshold = int(1 * factor)
    pad_bef = int(1 * factor)
    pad_aft = int(1 * factor)

    silence_bef = np.full_like(np.empty((pad_bef,), dtype=float), fill_value=0.0)
    silence_aft = np.full_like(np.empty((pad_aft,), dtype=float), fill_value=0.0)

    chunks = []

    for i, (st, et, sp) in enumerate(timeline_norm):
        start_idx = (st * factor)
        end_idx = (et * factor)

        # Ensure that end_idx does not exceed the length of the audio
        end_idx = min(end_idx, length)

        # Extract the chunk
        chunk = audio1[int(start_idx):round(end_idx)]

        chunk_length = len(chunk)
        if chunk_length < threshold:
            chunk = np.concatenate((silence_bef, audio1[round(start_idx):round(end_idx)]))
            chunk = np.concatenate((chunk, silence_aft))

            # print(f'i: {i}, chunk_length_before: {chunk_length}, after: {len(chunk)}')

        chunks.append(chunk)

    logging.info("Transcribing...")

    st = time.time()
    output_path = "output.wav"  # Replace with your desired output path
    text = list()
    for chunk in chunks:
        sf.write(output_path, chunk, sr)
        output = asr_model.transcribe(paths2audio_files=[output_path], batch_size=4)
        text.append({"text": output[0][0]})

    et = time.time()
    t = et - st
    os.remove(output_path)

    logging.info(f"Transcription time: {t:.3f} sec")

    return text, chunks
