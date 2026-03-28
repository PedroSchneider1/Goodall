# wave.py
import numpy as np

class Wave:
    """
    Represents a discrete-time waveform.
    This class was copied from Allen Downey 'Think DSP' book.
    It's used to represent audio data as a sequence of samples (ys) and their corresponding times (ts).
    The framerate attribute indicates the number of samples per second.
    This class was created to facilitate audio processing and feature extraction in the context of the project, allowing for easy manipulation and analysis of audio data.
    """

    def __init__(self, ys, ts=None, framerate=None):
        """
        Initializes the wave.

        ys: wave array
        ts: array of times
        framerate: samples per second
        """
        self.ys = np.asanyarray(ys)
        self.framerate = framerate if framerate is not None else 11025

        if ts is None:
            self.ts = np.arange(len(ys)) / self.framerate
        else:
            self.ts = np.asanyarray(ts)

    def __len__(self):
        return len(self.ys)

    @property
    def start(self):
        return self.ts[0]

    @property
    def end(self):
        return self.ts[-1]

    @property
    def duration(self):
        """Duration (property).

        returns: float duration in seconds
        """
        return len(self.ys) / self.framerate

    def find_index(self, t):
        """Find the index corresponding to a given time."""
        n = len(self)
        start = self.start
        end = self.end
        i = round((n - 1) * (t - start) / (end - start))
        return int(i)

    def segment(self, start=None, duration=None):
        """Extracts a segment.

        start: float start time in seconds
        duration: float duration in seconds

        returns: Wave
        """
        if start is None:
            start = self.ts[0]
            i = 0
        else:
            i = self.find_index(start)

        j = None if duration is None else self.find_index(start + duration)
        return self.slice(i, j)

    def slice(self, i, j):
        """Makes a slice from a Wave.

        i: first slice index
        j: second slice index
        """
        ys = self.ys[i:j].copy()
        ts = self.ts[i:j].copy()
        return Wave(ys, ts, self.framerate)