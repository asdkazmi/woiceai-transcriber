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
from pydub import AudioSegment
from pydub.silence import split_on_silence
from concurrent.futures import ThreadPoolExecutor

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
    if type(urls) == list:
        urls = [url for url in urls if url]
    else:
        logging.info("The provided variable is not a list.")
        return None, None

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


def retrieve_data_from_file(paths):
    all_audios = []
    diarization_list = []
    st = time.time()

    # Filter out empty strings from urls
    if type(paths) == list:
        paths = [path for path in paths if path]
    else:
        logging.info("The provided variable is not a list.")
        return None, None

    logging.info("Audio diarization in progress...")

    for path in paths:
        d = pipe(path, min_speakers=1, max_speakers=50)
        diarization_list.append(d)

        all_audios.append(librosa.load(path))

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
            os.remove(output_path)

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



def transcribe_without_diarization(paths, from_file):
    try:
        stt = time.time()
        transcriptions = []

        # Validate paths
        if isinstance(paths, list):
            paths = [path for path in paths if path]
        else:
            logging.error("The provided variable is not a list.")
            return None, None
        batch_size=5
        # Load audio files and split into chunk with timestamps
        logger.info(f"Loading audio and making chunks")
        all_chunks, all_timestamps = split_audio_with_timestamps(file_paths=paths, from_file=from_file)

        chunk_batches = [all_chunks[i:i + batch_size] for i in range(0, len(all_chunks), batch_size)]
        timestamp_batches = [all_timestamps[i:i + batch_size] for i in range(0, len(all_timestamps), batch_size)]
        logger.info(f"Transcribing")
        # Transcribe each batch in parallel
        for batch_idx, (batch, timestamps) in enumerate(zip(chunk_batches, timestamp_batches)):
            # Process and transcribe the current batch
            batch_transcriptions = process_batch(batch)  # Use first path as original_path

            # Combine transcriptions with their respective timestamps
            for transcription, (start, end) in zip(batch_transcriptions, timestamps):
                transcriptions.append({
                    "start": round(start / 1000, 2),  # Convert to seconds and round
                    "end": round(end / 1000, 2),
                    "text": transcription
                })

                # Remove temporary chunk file
                start_time = end_time

                # Clean up the chunk file
                os.remove(chunk_path)

        ett = time.time()
        t = ett - stt
        logging.info(f'Transcription time: {t:.3f} sec')

        return {
            "status": "completed",
            "transcription": transcriptions,
        }

    except Exception as e:
        logging.error(f"Error during transcription: {e}")
        return {"status": "error", "message": str(e)}


def transcribe_audio(chunk_path):
    try:
        # Actual transcription logic
        output = asr_model.transcribe(paths2audio_files=[chunk_path])
        transcription = output[0][0]  # Assuming the model returns a nested list
        return transcription
    except Exception as e:
        # Handle any exceptions during transcription
        logger.error(f"Error transcribing {chunk_path}: {e}")
        return f"Error: {e}"


def process_batch(batch_chunks):
    """
    Process a batch of audio chunks, export them as individual files, and transcribe them.
    """
    temp_files = []

    for idx, chunk in enumerate(batch_chunks):
        # Generate a custom file path based on the original file and chunk index
        chunk_path = f"output_segment_{idx + 1}.wav"

        # Export the chunk to a temporary file
        chunk.export(chunk_path, format="wav")
        temp_files.append(chunk_path)

    # Use ThreadPoolExecutor to process the files in parallel
    with ThreadPoolExecutor() as executor:
        transcriptions = list(executor.map(transcribe_audio, temp_files))
    [os.remove(temp_file) for temp_file in temp_files]
    return transcriptions


def split_audio_with_timestamps(file_paths,from_file, min_silence_len=500, silence_threshold=-40, keep_silence=500):
    all_chunks = []
    all_timestamps = []

    for file_path in file_paths:
        # Load audio file
        if from_file:
            audio = AudioSegment.from_file(file_path)
        else:
            response = requests.get(file_path)
            response.raise_for_status()
            audio = AudioSegment.from_file(BytesIO(response.content), format="wav")


        audio = audio.set_channels(1).set_frame_rate(16000)
        # Split audio into chunks based on silence
        chunks = split_on_silence(
            audio,
            min_silence_len=min_silence_len,
            silence_thresh=silence_threshold,
            keep_silence=keep_silence
        )
        logging.info(f"Detected {len(chunks)} chunks.")
        # Calculate start and end timestamps
        timestamps = []
        current_time = 0  # Start time in milliseconds
        for chunk in chunks:
            chunk_start = current_time
            chunk_end = current_time + len(chunk)
            timestamps.append((chunk_start, chunk_end))
            current_time = chunk_end

        all_chunks.extend(chunks)
        all_timestamps.extend(timestamps)

    return all_chunks, all_timestamps

# def transcribe_without_diarization(paths, from_file):
#     try:
#         stt = time.time()
#         logger.info(f"Transcribing : {paths}")
#         all_audios = []
#         transcriptions = []
#         combined_transcription=""
#         num_of_urls = len(paths)
#         if type(paths) == list:
#             paths = [path for path in paths if path]
#         else:
#             logging.info("The provided variable is not a list.")
#             return None, None
#
#         for path in paths:
#             if from_file:
#                 audio, sr = librosa.load(path, sr=16000, mono=True)
#                 all_audios.append((audio, sr, path))
#             else:
#                 response = requests.get(path)
#                 response.raise_for_status()
#                 audio = BytesIO(response.content)
#                 all_audios.append(librosa.load(audio))
#         for audio, sr, original_path in all_audios:
#             # Split the audio into 10-minute chunks
#             chunk_duration = 10 * 60 * sr  # 10 minutes in samples
#             for i in range(0, len(audio), chunk_duration):
#                 chunk = audio[i:i + chunk_duration]
#                 output_path = f"output_{original_path.split('/')[-1].replace('.wav', '')}_chunk_{i // chunk_duration + 1}.wav"
#                 sf.write(output_path, chunk, sr)
#
#                 # Transcribe the chunk
#                 output = asr_model.transcribe(paths2audio_files=[output_path], batch_size=1)
#                 print(output)
#                 transcription = output[0][0]
#                 combined_transcription += transcription + " "
#                 os.remove(output_path)
#         combined_transcription = combined_transcription.strip()
#         transcriptions.append([{
#             "text": combined_transcription,
#         }])
#         ett = time.time()
#         t = ett - stt
#         logging.info(f'Transcription time: {t:.3f} sec')
#
#         return {"status": "completed",
#                 "transcription": transcriptions
#                 }
#     except Exception as e:
#         print(e)