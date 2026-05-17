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
    target_scripts = list_supported_scripts()
    
    FIXED_SOURCE_SCRIPT = "tamil"

    with gr.Blocks(title="ASR Transcription & Transliteration") as demo:
        gr.Markdown(
            """
            # ASR-Based Transcription & Transliteration System
            Record live audio **or** upload a file to transcribe Tamil speech with
            **Whisper**, then transliterate to any supported Indic / Latin script
            using **indic-transliteration**.

            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                audio_input = gr.Audio(
                    label="🎙️ Record or Upload Tamil Audio",
                    type="filepath",
                    sources=["microphone", "upload"],
                )
                gr.Markdown(
                    "**ASR Input Language:** Tamil (fixed — model is `whisper-small` "
                    "fine-tuned with `LANGUAGE=ta`)"
                )
                target_script = gr.Dropdown(
                    choices=target_scripts,
                    value="latin",
                    label="Transliterate to Script",
                )
                use_chunked = gr.Checkbox(
                    label="Use chunked buffer pipeline (for long audio)",
                    value=False,
                )
                submit_btn = gr.Button(
                    "▶ Transcribe & Transliterate", variant="primary"
                )
                clear_btn = gr.Button("🗑️ Clear", variant="secondary")

            with gr.Column(scale=1):
                transcript_out = gr.Textbox(
                    label="ASR Transcript (Tamil)",
                    lines=6,
                    interactive=False,
                )
                translit_out = gr.Textbox(
                    label="Transliteration Output",
                    lines=6,
                    interactive=False,
                )
                info_out = gr.Textbox(
                    label="File Output Info",
                    lines=3,
                    interactive=False,
                )

        submit_btn.click(
            fn=lambda audio, tgt, chunked: process_audio(
                audio, FIXED_SOURCE_SCRIPT, tgt, chunked
            ),
            inputs=[audio_input, target_script, use_chunked],
            outputs=[transcript_out, translit_out, info_out],
        )

        clear_btn.click(
            fn=lambda: (None, "", "", ""),
            inputs=[],
            outputs=[audio_input, transcript_out, translit_out, info_out],
        )

        gr.Markdown(
            "---\n"
            "**Supported transliteration targets:** "
            + ", ".join(f"`{s}`" for s in target_scripts)
        )

    return demo