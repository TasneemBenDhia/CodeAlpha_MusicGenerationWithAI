import numpy as np
from music21 import stream, instrument, note, chord

def generate_music_sequence(model, raw_input, pitch_names, vocab_size, num_notes=200):
    """ Use weights to predict sequential notes """
    int_to_note = {num: note for num, note in enumerate(pitch_names)}
    
    start_index = np.random.randint(0, len(raw_input) - 1)
    pattern = list(raw_input[start_index])
    generated_notes = []

    print("Generating note sequence...")
    for _ in range(num_notes):
        prediction_input = np.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input / float(vocab_size)
        
        prediction = model.predict(prediction_input, verbose=0)
        index = np.argmax(prediction)
        
        generated_notes.append(int_to_note[index])
        pattern.append(index)
        pattern = pattern[1:]

    return generated_notes

def export_to_midi(generated_notes, output_filename="ai_melody.mid"):
    """ Map generated string representations back to a physical MIDI stream """
    offset = 0
    output_notes = []

    for pattern in generated_notes:
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            chord_notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                chord_notes.append(new_note)
            new_chord = chord.Chord(chord_notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)

        offset += 0.5

    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp=output_filename)
    print(f"Success! Output saved as {output_filename}")