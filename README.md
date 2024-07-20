# Music-Generation-with-TensorFlow
This repository contains a sample code for generating music using machine learning techniques with TensorFlow. The code processes MIDI files, trains a model, and generates new musical sequences.

## Initialization
First, import the necessary libraries and set the random seeds for reproducibility:
```
import collections
import datetime
import fluidsynth
import glob
import numpy as np
import pathlib
import pandas as pd
import pretty_midi
import seaborn as sns
import tensorflow as tf
import random
from IPython import display
from matplotlib import pyplot as plt
from typing import Optional

tf.random.set_seed(42)
np.random.seed(42)
_SAMPLING_RATE = 16000
```

## Dataset Preparation
Download and prepare the Maestro dataset:
```
data_dir = pathlib.Path('data/maestro-v2.0.0')
if not data_dir.exists():
    tf.keras.utils.get_file(
        'maestro-v2.0.0-midi.zip',
        origin='https://storage.googleapis.com/magentadata/datasets/maestro/v2.0.0/maestro-v2.0.0-midi.zip',
        extract=True,
        cache_dir='.', cache_subdir='data',
    )

search_pattern = str(data_dir / '**/*.mid*')
filenames = glob.glob(search_pattern)
print(f"Number of MIDI files found: {len(filenames)}")

sample_file = random.choice(filenames)
pm = pretty_midi.PrettyMIDI(sample_file)

def display_audio(pm: pretty_midi.PrettyMIDI, seconds=40):
    waveform = pm.fluidsynth(fs=_SAMPLING_RATE)
    waveform_short = waveform[:seconds*_SAMPLING_RATE]
    return display.Audio(waveform_short, rate=_SAMPLING_RATE)

display_audio(pm)
```

## Training
Extract notes from the MIDI files and train a neural network model. The complete training code is available in the provided notebook.

## Generation
Generate new music using the trained model:
```
def predict_next_note(
    notes: np.ndarray, 
    model: tf.keras.Model, 
    temperature: float = 1.0) -> tuple[int, float, float]:
    inputs = tf.expand_dims(notes, 0)
    predictions = model.predict(inputs)
    pitch_logits = predictions['pitch']
    step = predictions['step']
    duration = predictions['duration']
    
    pitch_logits /= temperature
    pitch = tf.random.categorical(pitch_logits, num_samples=1)
    pitch = tf.squeeze(pitch, axis=-1)
    duration = tf.squeeze(duration, axis=-1)
    step = tf.squeeze(step, axis=-1)
    
    step = tf.maximum(0, step)
    duration = tf.maximum(0, duration)
    
    return int(pitch), float(step), float(duration)

temperature = 2.0
num_predictions = 120

sample_notes = np.stack([raw_notes[key] for key in key_order], axis=1)
input_notes = (
    sample_notes[:seq_length] / np.array([vocab_size, 1, 1])
)

generated_notes = []
prev_start = 0
for _ in range(num_predictions):
    pitch, step, duration = predict_next_note(input_notes, model, temperature)
    start = prev_start + step
    end = start + duration
    input_note = (pitch, step, duration)
    generated_notes.append((*input_note, start, end))
    input_notes = np.delete(input_notes, 0, axis=0)
    input_notes = np.append(input_notes, np.expand_dims(input_note, 0), axis=0)
    prev_start = start

generated_notes = pd.DataFrame(
    generated_notes, columns=(*key_order, 'start', 'end')
)
```
Save the generated notes to a MIDI file and play it:
```
out_file = 'output.mid'
instrument = pm.instruments[0]
out_pm = notes_to_midi(
    generated_notes, out_file=out_file, instrument_name=pretty_midi.program_to_instrument_name(instrument.program)
)
display_audio(out_pm)
plot_piano_roll(generated_notes)
plot_distributions(generated_notes)
```

## Results
The generated MIDI file (output.mid) and various visualizations (piano roll, note distributions) are produced.
