import logging
import os
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa
import numpy as np

from models.model_config import ASR_MODEL_ID, SAMPLE_RATE, LANGUAGE, TASK

logger = logging.getLogger(__name__)


class ASRPipeline:
    

    def __init__(self, model_id: str = ASR_MODEL_ID):
        self.model_id = model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading ASR model '{model_id}' on {self.device}...")
        self.processor = WhisperProcessor.from_pretrained(model_id)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_id).to(self.device)
        self.model.eval()
        logger.info("ASR model loaded successfully.")

    def load_audio(self, audio_path: str) -> np.ndarray:
        # Load and resample audio to the required sample rate.
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        waveform, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
        logger.info(f"Loaded audio: {audio_path} | Duration: {len(waveform)/SAMPLE_RATE:.2f}s")
        return waveform

    def transcribe(self, audio_input) -> str:
        
        if isinstance(audio_input, str):
            waveform = self.load_audio(audio_input)
        elif isinstance(audio_input, np.ndarray):
            waveform = audio_input
        else:
            raise TypeError("audio_input must be a file path (str) or numpy array.")

        # Prepare input features
        inputs = self.processor(
            waveform,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
        ).input_features.to(self.device)

        # Force language and task tokens
        forced_decoder_ids = self.processor.get_decoder_prompt_ids(
            language=LANGUAGE, task=TASK
        )

        with torch.no_grad():
            predicted_ids = self.model.generate(
                inputs,
                forced_decoder_ids=forced_decoder_ids,
            )

        transcript = self.processor.batch_decode(
            predicted_ids, skip_special_tokens=True
        )[0].strip()

        logger.info(f"Transcription: {transcript[:80]}{'...' if len(transcript) > 80 else ''}")
        return transcript

    def transcribe_chunks(self, chunks: list) -> str:
        
        results = []
        for i, chunk in enumerate(chunks):
            logger.info(f"Transcribing chunk {i+1}/{len(chunks)}...")
            results.append(self.transcribe(chunk))
        return " ".join(results)
