import logging
import os
import gradio as gr
import numpy as np
import soundfile as sf

from asr_pipeline import ASRPipeline
from transliteration import transliterate_text, list_supported_scripts
from buffer_manager import AudioBufferManager
from utils import save_transcript, save_transliteration

logger = logging.getLogger(__name__)

_asr: ASRPipeline = None
_buffer: AudioBufferManager = None

OUTPUT_DIR_TRANSCRIPTS = os.path.join(
    os.path.dirname(__file__), "..", "outputs", "transcripts"
)
OUTPUT_DIR_TRANSLIT = os.path.join(
    os.path.dirname(__file__), "..", "outputs", "transliterations"
)


def _get_asr() -> ASRPipeline:
    global _asr
    if _asr is None:
        _asr = ASRPipeline()
    return _asr


def _get_buffer() -> AudioBufferManager:
    global _buffer
    if _buffer is None:
        _buffer = AudioBufferManager()
    return _buffer



def process_audio(
    audio_file,
    source_script: str,
    target_script: str,
    use_chunked: bool,
):
   
    if audio_file is None:
        return " No audio file provided.", "", ""

    asr = _get_asr()
    buf = _get_buffer()
    buf.clear()

    try:
        if use_chunked:
            buf.enqueue_file(audio_file)
            chunks = buf.drain()
            transcript = asr.transcribe_chunks(chunks)
        else:
            # Direct path: transcribe full file in one pass
            transcript = asr.transcribe(audio_file)
    except Exception as e:
        logger.error(f"ASR failed: {e}")
        return f" ASR error: {e}", "", ""

    try:
        transliterated = transliterate_text(transcript, source_script, target_script)
    except Exception as e:
        logger.error(f"Transliteration failed: {e}")
        transliterated = f"Transliteration error: {e}"

    transcript_path = save_transcript(transcript, OUTPUT_DIR_TRANSCRIPTS)
    translit_path = save_transliteration(transliterated, OUTPUT_DIR_TRANSLIT)

    info = (
        f"Transcript saved → {transcript_path}\n"
        f"Transliteration saved → {translit_path}"
    )
    return transcript, transliterated, info



def build_interface() -> gr.Blocks:
    scripts = list_supported_scripts()

    with gr.Blocks(title="ASR Transcription & Transliteration") as demo:
        gr.Markdown(
            """
            # ASR-Based Transcription & Transliteration System
            Upload an audio file to transcribe it with **Whisper** and optionally
            transliterate the output between Indic scripts using **indic-transliteration**.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                audio_input = gr.Audio(
                    label="Upload Audio File",
                    type="filepath",
                    sources=["upload", "microphone"],
                )
                source_script = gr.Dropdown(
                    choices=scripts,
                    value="tamil",
                    label="Source Script (ASR output language)",
                )
                target_script = gr.Dropdown(
                    choices=scripts,
                    value="latin",
                    label="Target Script (transliteration output)",
                )
                use_chunked = gr.Checkbox(
                    label="Use chunked buffer pipeline",
                    value=False,
                )
                submit_btn = gr.Button("▶ Transcribe & Transliterate", variant="primary")

            with gr.Column(scale=1):
                transcript_out = gr.Textbox(
                    label="ASR Transcript",
                    lines=6,
                    interactive=False,
                )
                translit_out = gr.Textbox(
                    label="Transliteration",
                    lines=6,
                    interactive=False,
                )
                info_out = gr.Textbox(
                    label="File Output Info",
                    lines=3,
                    interactive=False,
                )

        submit_btn.click(
            fn=process_audio,
            inputs=[audio_input, source_script, target_script, use_chunked],
            outputs=[transcript_out, translit_out, info_out],
        )

        gr.Markdown(
            "---\n"
            "**Supported scripts:** "
            + ", ".join(f"`{s}`" for s in scripts)
        )

    return demo
