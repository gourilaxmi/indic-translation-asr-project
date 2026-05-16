import logging
import queue
import threading
import numpy as np
import librosa

from models.model_config import SAMPLE_RATE, CHUNK_DURATION_S, BUFFER_MAXSIZE

logger = logging.getLogger(__name__)


class AudioBufferManager:

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        chunk_duration: float = CHUNK_DURATION_S,
        maxsize: int = BUFFER_MAXSIZE,
    ):
        self.sample_rate = sample_rate
        self.chunk_samples = int(chunk_duration * sample_rate)
        self._queue: queue.Queue = queue.Queue(maxsize=maxsize)
        self._lock = threading.Lock()
        logger.info(
            f"AudioBufferManager initialised | "
            f"chunk={chunk_duration}s ({self.chunk_samples} samples) | "
            f"maxsize={maxsize}"
        )

    def enqueue_file(self, audio_path: str) -> int:
        waveform, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
        chunks = self.split(waveform)
        for chunk in chunks:
            self.drop_old_chunk(chunk)
        logger.info(f"Enqueued {len(chunks)} chunk(s) from '{audio_path}'")
        return len(chunks)

    def enqueue_chunk(self, chunk: np.ndarray) -> None:
        if not isinstance(chunk, np.ndarray):
            raise TypeError("chunk must be a numpy ndarray.")
        self.drop_old_chunk(chunk)

    def dequeue(self, timeout: float = 2.0):
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def drain(self) -> list:
        chunks = []
        while not self._queue.empty():
            try:
                chunks.append(self._queue.get_nowait())
            except queue.Empty:
                break
        logger.info(f"Drained {len(chunks)} chunk(s) from buffer.")
        return chunks

    def is_empty(self) -> bool:
        return self._queue.empty()

    def size(self) -> int:
        return self._queue.qsize()

    def clear(self) -> None:
        with self._lock:
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break
        logger.info("Buffer cleared.")

    def split(self, waveform: np.ndarray) -> list:
        n = len(waveform)
        chunks = []
        for start in range(0, n, self.chunk_samples):
            chunk = waveform[start: start + self.chunk_samples]
            if len(chunk) < self.chunk_samples:
                chunk = np.pad(chunk, (0, self.chunk_samples - len(chunk)))
            chunks.append(chunk)
        return chunks

    def drop_old_chunk(self, chunk: np.ndarray) -> None:
        try:
            self._queue.put_nowait(chunk)
        except queue.Full:
            logger.warning("Buffer full — dropping oldest chunk to make room.")
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            self._queue.put_nowait(chunk)