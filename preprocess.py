import glob
import numpy as np
import pickle
from music21 import converter, instrument, note, chord
from tensorflow.keras.utils import to_categorical

def get_notes_from_midi():
    """ Extract all notes and chords from the midi files """
    notes = []
    for file in glob.glob("midi_songs/*.mid"):
        try:
            print(f"Parsing {file}...")
            midi = converter.parse(file)
            parts = instrument.partitionByInstrument(midi)
            if parts: 
                notes_to_parse = parts.parts[0].recurse()
            else: 
                notes_to_parse = midi.flat.notes

            for element in notes_to_parse:
                if isinstance(element, note.Note):
                    notes.append(str(element.pitch))
                elif isinstance(element, chord.Chord):
                    notes.append('.'.join(str(n) for n in element.normalOrder))
        except Exception as e:
            print(f"Failed to parse {file}: {e}")
            
    return notes

def prepare_sequences(sequence_length=50):
    """ Loads data, maps it to integers, saves metadata, and formats for LSTM """
    all_notes = get_notes_from_midi()
    
    # Extract vocabulary
    pitch_names = sorted(set(all_notes))
    vocab_size = len(pitch_names)
    note_to_int = {note: num for num, note in enumerate(pitch_names)}
    
    # CRITICAL: Save vocabulary mapping for generator script later
    with open('pitch_names.pkl', 'wb') as f:
        pickle.dump(pitch_names, f)

    network_input = []
    network_output = []

    for i in range(0, len(all_notes) - sequence_length, 1):
        sequence_in = all_notes[i:i + sequence_length]
        sequence_out = all_notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)

    # Reshape and Normalize
    X = np.reshape(network_input, (n_patterns, sequence_length, 1))
    X = X / float(vocab_size)
    
    # One-hot encode output
    y = to_categorical(network_output, num_classes=vocab_size)

    return X, y, vocab_size, network_input