import os
import sys
import argparse
import numpy as np
from music21 import converter, instrument, note, chord, stream
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense, Activation
from tensorflow.keras.utils import to_categorical

# ----------------------------------------------------------------------------
# Before running:
# 1. Ensure you have a "data" directory in the project root.
# 2. Inside "data", create subfolders named exactly: classical, jazz, rock, blues.
# 3. Place your .mid or .midi files in the appropriate genre subfolder.
#    e.g. data/jazz/song1.mid, data/jazz/song2.mid
# ----------------------------------------------------------------------------

SEQUENCE_LENGTH = 100
EPOCHS = 3
BATCH_SIZE = 64
GENERATE_NOTES = 500


def load_midi_files(data_dir):
    """
    Load all MIDI files from `data_dir` and extract notes/chords.
    """
    notes = []
    for file in os.listdir(data_dir):
        if file.lower().endswith(('.mid', '.midi')):
            midi = converter.parse(os.path.join(data_dir, file))
            parts = instrument.partitionByInstrument(midi)
            elements = parts.parts[0].recurse() if parts else midi.flat.notes
            for el in elements:
                if isinstance(el, note.Note):
                    notes.append(str(el.pitch))
                elif isinstance(el, chord.Chord):
                    notes.append('.'.join(str(n) for n in el.normalOrder))
    if not notes:
        print(f"No MIDI files found in {data_dir}.")
        sys.exit(1)
    return notes


def prepare_sequences(notes, seq_length):
    pitchnames = sorted(set(notes))
    note_to_int = {n: i for i, n in enumerate(pitchnames)}

    network_input, network_output = [], []
    for i in range(len(notes) - seq_length):
        seq_in = notes[i:i + seq_length]
        seq_out = notes[i + seq_length]
        network_input.append([note_to_int[n] for n in seq_in])
        network_output.append(note_to_int[seq_out])

    n_patterns = len(network_input)
    n_vocab = len(pitchnames)
    network_input = np.reshape(network_input, (n_patterns, seq_length, 1))
    network_input = network_input / float(n_vocab)
    network_output = to_categorical(network_output, num_classes=n_vocab)
    return network_input, network_output, n_vocab, pitchnames, note_to_int


def create_model(input_shape, n_vocab):
    model = Sequential([
        LSTM(512, input_shape=(input_shape[1], input_shape[2]), return_sequences=True),
        Dropout(0.3),
        LSTM(512, return_sequences=True),
        Dropout(0.3),
        LSTM(512),
        Dense(256),
        Dropout(0.3),
        Dense(n_vocab),
        Activation('softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    return model


def generate_notes(model, network_input, pitchnames, note_to_int, seq_length, n_generate):
    int_to_note = {i: n for n, i in note_to_int.items()}
    start_idx = np.random.randint(0, len(network_input) - 1)
    pattern = list((network_input[start_idx].flatten() * len(pitchnames)).astype(int))
    output = []
    for _ in range(n_generate):
        x = np.reshape(pattern, (1, seq_length, 1))
        x = x / float(len(pitchnames))
        pred = model.predict(x, verbose=0)
        idx = np.argmax(pred)
        output.append(int_to_note[idx])
        pattern.append(idx)
        pattern = pattern[1:]
    return output


def create_midi(prediction_output, output_file):
    offset = 0
    notes_list = []
    for pattern in prediction_output:
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = [note.Note(int(n)) for n in pattern.split('.')]
            for n in notes_in_chord:
                n.storedInstrument = instrument.Piano()
            chord_obj = chord.Chord(notes_in_chord)
            chord_obj.offset = offset
            notes_list.append(chord_obj)
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            notes_list.append(new_note)
        offset += 0.5
    midi_stream = stream.Stream(notes_list)
    midi_stream.write('midi', fp=output_file)
    print(f"MIDI file saved as {output_file}")


def parse_args():
    parser = argparse.ArgumentParser(description="LSTM-based MIDI music generator/fine-tuner")
    parser.add_argument('genre', choices=['classical','jazz','rock','blues'],
                        help="Genre to train/generate from")
    parser.add_argument('--retrain', action='store_true',
                        help="Retrain the model on this genre even if weights exist")
    return parser.parse_args()


def main():
    args = parse_args()
    # Use separate weight files per genre to avoid overwriting
    weight_file = f"weights_{args.genre}.weights.h5"

    data_dir = os.path.join('data', args.genre)
    if not os.path.isdir(data_dir):
        print(f"Data directory '{data_dir}' not found.")
        sys.exit(1)

    notes = load_midi_files(data_dir)
    network_input, network_output, n_vocab, pitchnames, note_to_int = prepare_sequences(notes, SEQUENCE_LENGTH)
    model = create_model(network_input.shape, n_vocab)

    if os.path.isfile(weight_file) and not args.retrain:
        print(f"Loading weights from {weight_file}...")
        model.load_weights(weight_file)
    else:
        print(f"Training model on genre: {args.genre}")
        model.fit(network_input, network_output, epochs=EPOCHS, batch_size=BATCH_SIZE)
        model.save_weights(weight_file)
        print(f"Weights saved to {weight_file}")

    print("Generating music...")
    output_notes = generate_notes(model, network_input, pitchnames, note_to_int, SEQUENCE_LENGTH, GENERATE_NOTES)
    output_file = f"{args.genre}_output.mid"
    create_midi(output_notes, output_file)

if __name__ == '__main__':
    main()
