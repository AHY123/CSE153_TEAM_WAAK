import pretty_midi
import os
import numpy as np
import json
from sklearn.model_selection import train_test_split

# Define constants for tokenization and representation
MIDI_DATA_PATH = "./data/nesmdb_midi/all"
OUTPUT_DIR = "./processed_data_pulse1_lead"
VOCAB_FILE = os.path.join(OUTPUT_DIR, "nes_pulse1_lead_vocab.json")
SEQUENCES_FILE = os.path.join(OUTPUT_DIR, "nes_pulse1_lead_sequences.npz")
TIME_QUANTIZATION_STEP = 1 / 120  # Corresponds to 120Hz, as suggested by NES-MDB
MAX_TIME_SHIFT_STEPS = int(5 / TIME_QUANTIZATION_STEP) # Max time shift of 5 seconds

# Special tokens
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"
PAD_TOKEN = "<PAD>"

# For focusing on pulse_1_lead
TARGET_INSTRUMENT_NAME = "pulse_1_lead"
CHANNEL_NAME = "P1L" # Representing Pulse 1 Lead

def get_midi_files(path):
    """Gets all MIDI files from the specified path."""
    midi_files = []
    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith(".mid") or file.endswith(".midi"):
                midi_files.append(os.path.join(root, file))
    return midi_files

def midi_to_event_sequence(midi_file_path):
    """
    Converts a single MIDI file to a sequence of symbolic events,
    focusing only on the 'pulse_1_lead' track.
    Events are: NOTE_ON_P1L_p, NOTE_OFF_P1L_p, TIME_SHIFT_t
    p: pitch (0-127)
    t: time shift in quantized steps
    """
    try:
        midi_data = pretty_midi.PrettyMIDI(midi_file_path)
    except Exception as e:
        print(f"Error parsing {midi_file_path}: {e}")
        return None

    events = []
    current_time_seconds = 0.0
    lead_pulse_instrument = None

    for instrument in midi_data.instruments:
        if instrument.name.lower() == TARGET_INSTRUMENT_NAME.lower():
            lead_pulse_instrument = instrument
            break
    
    if not lead_pulse_instrument:
        # print(f"Warning: '{TARGET_INSTRUMENT_NAME}' track not found in {os.path.basename(midi_file_path)}.")
        # print(f"Available instrument names: {[inst.name for inst in midi_data.instruments]}")
        return None # Or an empty list if you prefer to generate an empty sequence

    timed_notes = []
    for note in lead_pulse_instrument.notes:
        # Ensure pitch is within MIDI range if necessary, though pretty_midi handles it.
        # Velocity is ignored for this representation.
        timed_notes.append({'time': note.start, 'type': 'on', 'pitch': note.pitch})
        timed_notes.append({'time': note.end, 'type': 'off', 'pitch': note.pitch})

    if not timed_notes:
        return [] # No notes found in the lead track

    timed_notes.sort(key=lambda x: x['time'])

    for note_event in timed_notes:
        event_time_seconds = note_event['time']
        time_delta_seconds = event_time_seconds - current_time_seconds

        if time_delta_seconds > 1e-5: # Only add time shift if it's significant
            time_shift_steps = round(time_delta_seconds / TIME_QUANTIZATION_STEP)
            if time_shift_steps > 0:
                capped_time_shift_steps = min(time_shift_steps, MAX_TIME_SHIFT_STEPS)
                # Distribute large time shifts into multiple MAX_TIME_SHIFT_STEPS tokens
                num_max_shifts = capped_time_shift_steps // MAX_TIME_SHIFT_STEPS
                for _ in range(num_max_shifts):
                    events.append(f"TIME_SHIFT_{MAX_TIME_SHIFT_STEPS}")
                remaining_shift = capped_time_shift_steps % MAX_TIME_SHIFT_STEPS
                if remaining_shift > 0:
                    events.append(f"TIME_SHIFT_{remaining_shift}")
        
        if note_event['type'] == 'on':
            events.append(f"NOTE_ON_{CHANNEL_NAME}_{note_event['pitch']}")
        else:
            events.append(f"NOTE_OFF_{CHANNEL_NAME}_{note_event['pitch']}")
        
        current_time_seconds = event_time_seconds
        
    return events

def build_vocabulary(all_event_sequences):
    """Builds a vocabulary from all event sequences."""
    unique_tokens = set([PAD_TOKEN, SOS_TOKEN, EOS_TOKEN])
    for seq in all_event_sequences:
        if seq: # Ensure sequence is not None
            for token in seq:
                unique_tokens.add(token)
    
    # Add all possible TIME_SHIFT tokens to ensure they are in vocab
    for i in range(1, MAX_TIME_SHIFT_STEPS + 1):
        unique_tokens.add(f"TIME_SHIFT_{i}")

    # Add all possible NOTE_ON/OFF tokens for the target channel
    for pitch in range(128):
        unique_tokens.add(f"NOTE_ON_{CHANNEL_NAME}_{pitch}")
        unique_tokens.add(f"NOTE_OFF_{CHANNEL_NAME}_{pitch}")
        
    vocab = {token: i for i, token in enumerate(sorted(list(unique_tokens)))}
    return vocab

def tokenize_sequences(all_event_sequences, vocab):
    """Converts event sequences to integer token sequences using the vocabulary."""
    tokenized_sequences = []
    for seq in all_event_sequences:
        if not seq: # Skip None sequences (e.g. if track wasn't found)
            continue
        tokenized_seq = [vocab[SOS_TOKEN]] + [vocab[token] for token in seq if token in vocab] + [vocab[EOS_TOKEN]]
        tokenized_sequences.append(np.array(tokenized_seq, dtype=np.int32))
    return tokenized_sequences

def main():
    """Main preprocessing script."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print("Getting MIDI files...")
    midi_files = get_midi_files(MIDI_DATA_PATH)
    if not midi_files:
        print(f"No MIDI files found in {MIDI_DATA_PATH}. Exiting.")
        return
    print(f"Found {len(midi_files)} MIDI files.")

    all_event_sequences = []
    processed_count = 0
    not_found_count = 0
    error_count = 0
    print("Converting MIDI files to event sequences (focusing on 'pulse_1_lead')...")
    for i, midi_file in enumerate(midi_files):
        # if (i+1) % 50 == 0:
        #     print(f"Processing file {i+1}/{len(midi_files)}: {os.path.basename(midi_file)}")
        event_sequence = midi_to_event_sequence(midi_file)
        if event_sequence is None:
            # This can happen if parsing failed or target track wasn't found
            if os.path.exists(midi_file): # Check if it was a parsing error or track not found
                 try:
                    pm = pretty_midi.PrettyMIDI(midi_file)
                    target_found = any(inst.name.lower() == TARGET_INSTRUMENT_NAME.lower() for inst in pm.instruments)
                    if not target_found:
                        not_found_count +=1
                        # print(f"  '{TARGET_INSTRUMENT_NAME}' not found in {os.path.basename(midi_file)}")
                 except Exception:
                    error_count +=1 # Likely parsing error
            else:
                error_count +=1 # File doesn't exist, though get_midi_files should prevent this
        elif event_sequence: # Non-empty sequence from target track
            all_event_sequences.append(event_sequence)
            processed_count += 1
        # else: sequence is empty list, means target track had no notes
        # We can choose to add these empty sequences or not.
        # For now, only adding sequences with actual events.

        if (i + 1) % 200 == 0: # Log progress
             print(f"  Processed {i+1} files. Found target track in {processed_count}. Target not found in {not_found_count}. Errors: {error_count}")
    
    print(f"Finished processing all files.")
    print(f"Successfully extracted '{TARGET_INSTRUMENT_NAME}' sequences from {processed_count} MIDI files.")
    print(f"'{TARGET_INSTRUMENT_NAME}' track not found in {not_found_count} files.")
    print(f"Could not parse or access {error_count} files.")

    if not all_event_sequences:
        print(f"No valid event sequences for '{TARGET_INSTRUMENT_NAME}' could be generated. Exiting.")
        return
    
    print("Building vocabulary...")
    vocab = build_vocabulary(all_event_sequences)
    print(f"Vocabulary size: {len(vocab)}")
    with open(VOCAB_FILE, 'w') as f:
        json.dump(vocab, f, indent=2)
    print(f"Vocabulary saved to {VOCAB_FILE}")

    print("Tokenizing sequences...")
    tokenized_sequences = tokenize_sequences(all_event_sequences, vocab)
    
    print(f"Saving {len(tokenized_sequences)} tokenized sequences...")
    np.savez_compressed(SEQUENCES_FILE, all_sequences=tokenized_sequences)
    print(f"All tokenized sequences for '{TARGET_INSTRUMENT_NAME}' saved to {SEQUENCES_FILE}")
    print("Preprocessing complete.")

if __name__ == "__main__":
    main() 