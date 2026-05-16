"""
test_pipeline.py - Unit and integration tests for the ASR + Transliteration pipeline.
Run with:  pytest tests/test_pipeline.py -v
"""

import os
import sys
import numpy as np
import pytest

# Make app/ importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))

from buffer_manager import AudioBufferManager
from transliteration import transliterate_text, list_supported_scripts
from utils import save_transcript, save_transliteration, list_output_files
from model_config import SAMPLE_RATE, CHUNK_DURATION_S


# ---------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------

@pytest.fixture
def buffer():
    return AudioBufferManager(chunk_duration=1.0, maxsize=10)


@pytest.fixture
def sample_waveform():
    """One second of silence at SAMPLE_RATE."""
    return np.zeros(SAMPLE_RATE, dtype=np.float32)


@pytest.fixture
def tmp_output(tmp_path):
    return str(tmp_path)


# ---------------------------------------------------------------
# Buffer tests
# ---------------------------------------------------------------

class TestAudioBufferManager:

    def test_enqueue_and_drain(self, buffer, sample_waveform):
        buffer.enqueue_chunk(sample_waveform)
        chunks = buffer.drain()
        assert len(chunks) == 1
        assert isinstance(chunks[0], np.ndarray)

    def test_drain_empty(self, buffer):
        assert buffer.drain() == []

    def test_is_empty(self, buffer, sample_waveform):
        assert buffer.is_empty()
        buffer.enqueue_chunk(sample_waveform)
        assert not buffer.is_empty()

    def test_clear(self, buffer, sample_waveform):
        buffer.enqueue_chunk(sample_waveform)
        buffer.clear()
        assert buffer.is_empty()

    def test_overflow_drops_oldest(self):
        small_buf = AudioBufferManager(chunk_duration=0.1, maxsize=2)
        a = np.ones(1600, dtype=np.float32) * 1.0
        b = np.ones(1600, dtype=np.float32) * 2.0
        c = np.ones(1600, dtype=np.float32) * 3.0
        small_buf.enqueue_chunk(a)
        small_buf.enqueue_chunk(b)
        small_buf.enqueue_chunk(c)   # should drop 'a'
        chunks = small_buf.drain()
        assert len(chunks) == 2

    def test_split_pads_last_chunk(self, buffer):
        # Waveform that is not an exact multiple of chunk_samples
        odd = np.zeros(int(SAMPLE_RATE * 1.5), dtype=np.float32)
        chunks = buffer._split(odd)
        for chunk in chunks:
            assert len(chunk) == buffer.chunk_samples

    def test_enqueue_wrong_type(self, buffer):
        with pytest.raises(TypeError):
            buffer.enqueue_chunk("not_an_array")

    def test_size(self, buffer, sample_waveform):
        assert buffer.size() == 0
        buffer.enqueue_chunk(sample_waveform)
        assert buffer.size() == 1


# ---------------------------------------------------------------
# Transliteration tests
# ---------------------------------------------------------------

class TestTransliteration:

    def test_list_scripts_nonempty(self):
        scripts = list_supported_scripts()
        assert len(scripts) > 0
        assert "tamil" in scripts
        assert "latin" in scripts

    def test_tamil_to_itrans(self):
        # Basic Tamil vowel: அ -> a (ITRANS)
        result = transliterate_text("அ", "tamil", "itrans")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_empty_string(self):
        result = transliterate_text("", "tamil", "latin")
        assert result == ""

    def test_whitespace_only(self):
        result = transliterate_text("   ", "tamil", "latin")
        assert result == ""

    def test_invalid_script_raises(self):
        with pytest.raises(ValueError):
            transliterate_text("test", "nonexistent_script", "tamil")

    def test_same_source_target(self):
        # Transliterating to the same script should be a no-op
        text = "hello"
        result = transliterate_text(text, "itrans", "itrans")
        assert isinstance(result, str)


# ---------------------------------------------------------------
# Utility tests
# ---------------------------------------------------------------

class TestUtils:

    def test_save_and_list_transcript(self, tmp_output):
        path = save_transcript("Hello Tamil", tmp_output)
        assert os.path.exists(path)
        files = list_output_files(tmp_output)
        assert path in files

    def test_save_and_list_transliteration(self, tmp_output):
        path = save_transliteration("வணக்கம்", tmp_output)
        assert os.path.exists(path)
        content = open(path, encoding="utf-8").read()
        assert "வணக்கம்" in content

    def test_list_output_files_empty_dir(self, tmp_output):
        files = list_output_files(tmp_output)
        assert files == []

    def test_list_output_files_missing_dir(self):
        files = list_output_files("/nonexistent/path")
        assert files == []
