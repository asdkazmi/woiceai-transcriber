import ffmpeg
import os
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def convert_audio_to_wav(input_file):

    directory = os.path.dirname(input_file)
    filename_without_ext = os.path.splitext(os.path.basename(input_file))[0]
    output_file = os.path.join(directory, f"{filename_without_ext}.wav")

    try:
        # Convert the audio file to WAV
        ffmpeg.input(input_file).output(output_file).run()
        logger.info(f"The file is successfully converted to : {output_file}")
        return output_file
    except ffmpeg.Error as e:
        logger.info(f"Error during conversion: {e}")