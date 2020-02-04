import matplotlib.pyplot as plt
import numpy as np
import music21 as m21
import tempfile
import fractions
import librosa
import librosa.display
import json
import peakutils
import base64
import pandas as pd
import matplotlib as mpl
import soundfile as sf

from itertools import groupby
from midi2audio import FluidSynth
from IPython.display import display, Audio, Markdown, Image, YouTubeVideo, HTML, IFrame
from IPython.display import Latex
from collections.abc import Iterable
from scipy.signal import lfilter, firwin
from scipy.signal.windows import tukey
from IPython import embed
from pathlib import Path
from scipy.interpolate import interp2d
from matplotlib.colors import hsv_to_rgb


def log_freqs_interp(X, log_freq_bins, fft_frequencies, frames_time):
    x = frames_time
    y = fft_frequencies
    f2d = interp2d(x, y, X)
    log_fft_frequencies = np.logspace(np.log10(fft_frequencies[1] / 2), np.log10(fft_frequencies[-1]), log_freq_bins)
    return f2d(frames_time, log_fft_frequencies), log_fft_frequencies


def magphase_hsv(M, ph):
    hue = ph / 2 / np.pi + 0.5
    Pdb = 20 * np.log10(M) + 100
    Pdb = Pdb.clip(0, None)
    Pdb = Pdb / Pdb.max()
    X = np.zeros((M.shape[0], M.shape[1], 3))
    X[:, :, 0] = hue
    X[:, :, 1] = Pdb
    X[:, :, 2] = Pdb**4
    return hsv_to_rgb(X)


def AudioPlayer(data, rate, norm=True, name=None, dir='./temp'):
    if name is None:
        with tempfile.NamedTemporaryFile(dir=dir, prefix='audio_', suffix='.wav', delete=False) as fp:
            name = fp.name
            name = Path(name).name
    else:
        name += '.wav'
    name = str(Path(dir, name))
    if norm:
        data /= np.max(np.abs(data))
        data *= 0.99
    sf.write(name, data, rate, subtype='PCM_16')
    return Audio(url=name, embed=False)


def spectrogram(y, fmin=0, fmax=None, sr=22050):
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    librosa.display.specshow(D, y_axis='linear', sr=sr, x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.ylim(fmin, fmax)


class Fr(fractions.Fraction):
    def __repr__(self):
        return '(%s/%s)' % (self.numerator, self.denominator)


def nearest_reduced_ratio(x, limit=0):
    fra = []
    if not isinstance(x, Iterable):
        x = [x]
    for x_ in x:
        if limit < 1:
            fra.append(Fr(x_))
        else:
            fra.append(Fr(x_).limit_denominator(limit))
    return fra


def farey(n, length=False):
    if not length:
        return [Fr(0, 1)] + sorted({Fr(m, k) for k in range(1, n + 1) for m in range(1, k + 1)})
    else:
        # return 1         +    len({Fr(m, k) for k in range(1, n+1) for m in range(1, k+1)})
        return (n * (n + 3)) // 2 - sum(farey(n // k, True) for k in range(2, n + 1))


def secuencia(notas, duraciones=None, velocidades=None, instrumento=None, chromatic=False,
              bpm=60, octava=4, rest_char='', scale=m21.scale.MajorScale('C')):
    if scale is not None:
        notas_ = []
        for n in notas:
            if isinstance(n, Iterable):
                chord_notes = [scale.pitchFromDegree(n_ + 1).transpose(12 * (octava - 4 + n_ // (len(scale.pitches) - 1))) for n_ in n]
                notas_.append(chord_notes)
            else:
                n_ = scale.pitchFromDegree(n + 1).transpose(12 * (octava - 4 + n // (len(scale.pitches) - 1)))
                notas_.append(n_)
        notas = notas_
    if duraciones is None:
        duraciones = [1] * len(notas)
    elif isinstance(duraciones, (int, float)):
        duraciones = [duraciones] * len(notas)
    if instrumento is None:
        instrumento = m21.instrument.Piano()
    s = m21.stream.Stream()
    s.append(m21.tempo.MetronomeMark(number=bpm))
    s.append(instrumento)
    for n, d in zip(notas, duraciones):
        if n == rest_char:
            s.append(m21.note.Rest(n, duration=m21.duration.Duration(d)))
        elif isinstance(n, Iterable):
            chord_notes = [m21.note.Note(n_, duration=m21.duration.Duration(d)) for n_ in n]
            s.append(m21.chord.Chord(chord_notes))
        else:
            s.append(m21.note.Note(n, duration=m21.duration.Duration(d)))
    if velocidades is None:
        velocidades = [127] * len(notas)
    elif isinstance(velocidades, (int, float)):
        velocidades = [velocidades] * len(notas)
    for n, v in zip(s.flat.notes, velocidades):
        if n != rest_char:
            n.volume.velocity = v
    sc = m21.stream.Score()
    sc.insert(s)
    return sc


def to_wav(self, dir=None):
    with tempfile.NamedTemporaryFile(suffix='.mid') as fp1:
        self.write('midi', fp=fp1.name)
        fs = FluidSynth('/usr/share/sounds/sf2/FluidR3_GM.sf2')
        with tempfile.NamedTemporaryFile(suffix='.wav', dir=dir, delete=False) as fp2:
            fs.midi_to_audio(fp1.name, fp2.name)
            return fp2.name


def play(self):
    filename = self.to_wav()
    return Audio(filename=filename)


us = m21.environment.UserSettings()
us['musicxmlPath'] = '/usr/bin/musescore'
us['musescoreDirectPNGPath'] = '/usr/bin/musescore'

m21.stream.Score.to_wav = to_wav
m21.stream.Score.play = play


def compress(uncompressed):
    """Compress a string to a list of output symbols."""

    # Build the dictionary.
    dict_size = 256
    dictionary = dict((chr(i), i) for i in range(dict_size))
    # in Python 3: dictionary = {chr(i): i for i in range(dict_size)}

    w = ""
    result = []
    for c in uncompressed:
        wc = w + c
        if wc in dictionary:
            w = wc
        else:
            result.append(dictionary[w])
            # Add wc to the dictionary.
            dictionary[wc] = dict_size
            dict_size += 1
            w = c

    # Output the code for w.
    if w:
        result.append(dictionary[w])
    return result


def decompress(compressed):
    """Decompress a list of output ks to a string."""
    from io import StringIO

    # Build the dictionary.
    dict_size = 256
    dictionary = dict((i, chr(i)) for i in range(dict_size))
    # in Python 3: dictionary = {i: chr(i) for i in range(dict_size)}

    # use StringIO, otherwise this becomes O(N^2)
    # due to string concatenation in a loop
    result = StringIO()
    w = chr(compressed.pop(0))
    result.write(w)
    for k in compressed:
        if k in dictionary:
            entry = dictionary[k]
        elif k == dict_size:
            entry = w + w[0]
        else:
            raise ValueError('Bad compressed k: %s' % k)
        result.write(entry)

        # Add w+entry[0] to the dictionary.
        dictionary[dict_size] = w + entry[0]
        dict_size += 1

        w = entry
    return result.getvalue()


def tilt(tilt_code, with_output=False):
    tilt_code = f"o = {tilt_code}"
    encoded_code = base64.standard_b64encode(json.dumps(compress(tilt_code))[1:-1].encode()).decode('ascii')
    display(IFrame(f"https://munshkr.gitlab.io/tilt/#v=1&c={encoded_code}", width="940", height="200"))


def spiral(N, M):
    x, y = 0, 0
    dx, dy = 0, -1

    for dumb in range(N * M):
        if np.abs(x) == np.abs(y) and [dx, dy] != [1, 0] or x > 0 and y == 1 - x:
            dx, dy = -dy, dx            # corner, change direction

        if abs(x) > N / 2 or abs(y) > M / 2:    # non-square
            dx, dy = -dy, dx            # change direction
            x, y = -y + dx, x + dy          # jump

        yield x, y
        x, y = x + dx, y + dy


def antialiasing(x, n=20):
    b, a = firwin(n + 1, 0.8, window='hamming'), 1.
    return lfilter(b, a, x)
