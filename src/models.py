# from transformers import pipeline
import logging
import os

import nemo.collections.asr as nemo_asr
import torch
from dotenv import load_dotenv
from pyannote.audio import Pipeline

load_dotenv()

# Configure the logging system
logger = logging.getLogger()
logger.setLevel(logging.INFO)

device = "cuda" if torch.cuda.is_available() else "cpu"

logging.info(f'Loading speaker-diarization model... on {device}')

speaker_diarization_model = "/pyannote/speaker-diarization-3.1/config.yaml"
pipe = Pipeline.from_pretrained(speaker_diarization_model)

pipe = pipe.to(torch.device(device))

logging.info(f'Loading Speech recognition model... on {device}')
if os.path.exists('nvidia/parakeet-rnnt-0.6b/parakeet-rnnt-0.6b.nemo'):  # For use inside docker
    asr_model = nemo_asr.models.EncDecRNNTBPEModel.restore_from(
        'nvidia/parakeet-rnnt-0.6b/parakeet-rnnt-0.6b.nemo')
else:
    asr_model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(model_name="nvidia/parakeet-rnnt-0.6b")
# asr_model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(model_name="nvidia/parakeet-rnnt-1.1b")

# whisper = pipeline('automatic-speech-recognition', model="distil-whisper/distil-large-v2", device=device)

logging.info('...complete')
