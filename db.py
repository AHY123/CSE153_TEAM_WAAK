# %%
import pandas as pd
import mido
import pretty_midi
from collections import defaultdict
import math
import numpy as np


# %%
df = pd.read_csv('data/all_more_than_20.csv')
file_path = df[["file_name"]].assign(p='data/nesmdb_midi/all/' + df['file_name'])['p'].to_list()
file_path

# %%
midi_file = mido.MidiFile(file_path[0])
midi_file

# %%
def parse_midi_basic_info(midi_path):
    """
    Parses a MIDI file and extracts basic information and track details.
    """
    try:
        mid = mido.MidiFile(midi_path)
    except Exception as e:
        print(f"Error loading MIDI file {midi_path}: {e}")
        return None

    if mid.type != 1:
        print(f"Warning: Expected Type 1 MIDI, got Type {mid.type} for {midi_path}")
        # You might choose to skip or handle differently

    print(f"\n--- Processing: {midi_path} ---")
    print(f"Type: {mid.type}")
    print(f"Ticks per beat: {mid.ticks_per_beat}")
    print(f"Number of tracks: {len(mid.tracks)}")

    track_info = []
    for i, track in enumerate(mid.tracks):
        info = {
            "index": i,
            "name": "Unnamed",
            "program": 0, # Default program
            "message_count": len(track)
        }
        for msg in track: # Look for initial track name and program change
            if msg.is_meta and msg.type == 'track_name':
                info["name"] = msg.name
            if msg.type == 'program_change':
                info["program"] = msg.program
                break # Often the first program change defines the track's main instrument
        track_info.append(info)
        print(f"  Track {i}: Name='{info['name']}', Program={info['program']}, Messages={info['message_count']}")
    return mid, track_info

# --- Step 2: Part/Instrument Role Identification ---
def identify_track_roles(track_info_list, mid_file_path=""):
    """
    Identifies roles for each track based on track name and program change.
    THIS FUNCTION REQUIRES CUSTOMIZATION BASED ON YOUR DATASET.
    """
    roles = {} # track_index -> "ROLE_NAME"
    print(f"\nIdentifying roles for {mid_file_path}:")

    # --- !!! CUSTOMIZE THIS LOGIC EXTENSIVELY !!! ---
    # Example rules (very basic, you need to analyze your dataset):
    for track_info in track_info_list:
        track_idx = track_info["index"]
        track_name = track_info["name"].lower()
        program = track_info["program"]

        # These are placeholder examples.
        if "drum" in track_name or "noise" in track_name or "perc" in track_name or (program >= 112 and program <= 119): # GM Percussion patches
            roles[track_idx] = "NOISE"
        elif "bass" in track_name or (program >= 32 and program <= 39): # GM Bass patches
            roles[track_idx] = "BASS"
        elif "lead" in track_name or "pulse" in track_name or "melody" in track_name or (program >= 80 and program <= 87): # GM Lead patches
            roles[track_idx] = "LEAD_PULSE1"
        elif "harmony" in track_name or "chord" in track_name or "square" in track_name or (program >= 0 and program <= 15) : # GM Piano / Chromatic Perc / Organ
             # This is a very broad category, you might want more specific pulse/square wave roles
            roles[track_idx] = "HARMONY_PULSE2"
        else:
            roles[track_idx] = "OTHER" # Or skip tracks with role "OTHER" later

        print(f"  Track {track_idx} (Name: '{track_info['name']}', Prog: {program}) assigned role: {roles[track_idx]}")
    # --- !!! END CUSTOMIZATION SECTION !!! ---

    return roles

# --- Utility to extract notes from tracks ---
def extract_notes_from_tracks(mid, track_roles):
    """
    Extracts note events from each track, associating them with their identified role.
    Converts note_on/note_off pairs into (pitch, velocity, start_tick, end_tick, role) tuples.
    """
    notes_by_role = defaultdict(list)
    ticks_per_beat = mid.ticks_per_beat

    for track_idx, track in enumerate(mid.tracks):
        current_time_ticks = 0
        active_notes = {} # pitch -> (start_tick, velocity)
        role = track_roles.get(track_idx, "OTHER") # Get assigned role

        if role == "OTHER": # Optionally skip tracks not assigned a useful role
            # print(f"  Skipping track {track_idx} with role OTHER for note extraction.")
            continue

        for msg in track:
            current_time_ticks += msg.time # Accumulate delta time in ticks

            if msg.type == 'note_on' and msg.velocity > 0:
                active_notes[msg.note] = (current_time_ticks, msg.velocity)
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                if msg.note in active_notes:
                    start_tick, velocity = active_notes.pop(msg.note)
                    end_tick = current_time_ticks
                    if end_tick > start_tick: # Ensure duration is positive
                         notes_by_role[role].append({
                            "pitch": msg.note,
                            "velocity": velocity,
                            "start_tick": start_tick,
                            "end_tick": end_tick,
                            "role": role # Keep track of the role for this note
                        })
    return notes_by_role


# --- Step 3: Monophony Enforcement ---
def enforce_monophony(note_list):
    """
    Enforces monophony for a list of notes from a single part/role.
    Assumes note_list is sorted by start_tick.
    Modifies notes in place by adjusting end_tick.
    """
    if not note_list:
        return []

    # Sort by start_tick, then by pitch (as a tie-breaker, though less critical here)
    sorted_notes = sorted(note_list, key=lambda x: (x["start_tick"], x["pitch"]))
    
    monophonic_notes = []
    if not sorted_notes:
        return monophonic_notes

    monophonic_notes.append(dict(sorted_notes[0])) # Start with the first note

    for i in range(1, len(sorted_notes)):
        current_note = dict(sorted_notes[i])
        last_added_note = monophonic_notes[-1]

        # If the current note starts before the last added note ends,
        # truncate the last added note.
        if current_note["start_tick"] < last_added_note["end_tick"]:
            last_added_note["end_tick"] = current_note["start_tick"]
        
        # Add the current note if its duration is positive after potential truncation of previous
        if last_added_note["end_tick"] > last_added_note["start_tick"]: # Ensure previous note is still valid
            monophonic_notes.append(current_note)
        elif monophonic_notes: # If previous note became zero-duration, replace it if this one is valid
             monophonic_notes[-1] = current_note
        else: # If list was empty (should not happen if sorted_notes not empty)
            monophonic_notes.append(current_note)


    # Final pass to remove zero-duration notes that might have resulted
    return [note for note in monophonic_notes if note["end_tick"] > note["start_tick"]]


# --- Step 4: Velocity Processing & Quantization ---
def quantize_velocity(velocity, num_bins=4):
    """
    Quantizes MIDI velocity (0-127) into a smaller number of bins.
    THIS IS A BASIC EXAMPLE. You might want role-specific bins.
    """
    if velocity == 0: return 0 # Typically note_off
    # Ensure velocity is within 1-127 for binning
    velocity = max(1, min(127, velocity))
    bin_size = 127 / num_bins
    # Bin index (0 to num_bins-1)
    quantized_bin = int((velocity -1) / bin_size)
    return quantized_bin # Or return a representative value for the bin, e.g., bin_center


# --- Step 5: Tempo, Key, and Time Signature Handling ---
def extract_meta_info(mid):
    """
    Extracts tempo changes, key signatures, and time signatures.
    Decision on how to USE this info (normalize, tokenize, ignore) is separate.
    """
    tempos = [] # List of (tick, tempo_value_microseconds_per_beat)
    key_signatures = [] # List of (tick, key_name)
    time_signatures = [] # List of (tick, numerator, denominator, clocks_per_click, notated_32nd_notes_per_quarter)

    # Meta messages are often in track 0, but can be in others for Type 1.
    # We need to iterate all tracks and maintain absolute time.
    for track_idx, track in enumerate(mid.tracks):
        current_time_ticks = 0
        for msg in track:
            current_time_ticks += msg.time
            if msg.is_meta:
                if msg.type == 'set_tempo':
                    tempos.append({"tick": current_time_ticks, "tempo_us_per_beat": msg.tempo})
                elif msg.type == 'key_signature':
                    key_signatures.append({"tick": current_time_ticks, "key": msg.key})
                elif msg.type == 'time_signature':
                    time_signatures.append({
                        "tick": current_time_ticks,
                        "numerator": msg.numerator,
                        "denominator": msg.denominator,
                        "clocks_per_click": msg.clocks_per_click, # MIDI clocks per metronome click
                        "notated_32nd_notes_per_quarter": msg.notated_32nd_notes_per_beat # Number of 32nd notes in a MIDI quarter note
                    })
    
    # Sort by tick time as messages can come from different tracks
    tempos.sort(key=lambda x: x["tick"])
    key_signatures.sort(key=lambda x: x["tick"])
    time_signatures.sort(key=lambda x: x["tick"])

    print("\nMeta Information:")
    if not tempos: print("  No tempo changes found. Assume default 120 BPM (500000 us/beat).")
    else: print(f"  Tempos: {tempos}")
    
    if not key_signatures: print("  No key signatures found.")
    else: print(f"  Key Signatures: {key_signatures}")

    if not time_signatures: print("  No time signatures found. Assume default 4/4.")
    else: print(f"  Time Signatures: {time_signatures}")
        
    return tempos, key_signatures, time_signatures


# --- Main Processing Function to Orchestrate Steps ---
def preprocess_midi_file(midi_path, num_velocity_bins=4):
    """
    Orchestrates the preprocessing steps for a single MIDI file.
    """
    # Step 1
    mid, track_info_list = parse_midi_basic_info(midi_path)
    if mid is None:
        return None

    # Step 2
    # !!! CRITICAL: You MUST customize identify_track_roles for your data !!!
    track_roles = identify_track_roles(track_info_list, midi_path)
    
    # Utility: Extract notes based on these roles
    notes_by_role_raw = extract_notes_from_tracks(mid, track_roles)

    # Step 3 & 4 (Applied per role)
    processed_notes_all_roles = []
    roles_to_make_monophonic = ["LEAD_PULSE1", "HARMONY_PULSE2", "BASS"] # Customize this list!

    print("\nProcessing notes per role:")
    for role, notes in notes_by_role_raw.items():
        print(f"  Role: {role}, Raw note count: {len(notes)}")
        if not notes:
            continue

        # Step 3: Monophony
        current_role_notes = notes # Use raw notes for the role
        if role in roles_to_make_monophonic:
            print(f"    Enforcing monophony for role: {role}")
            current_role_notes = enforce_monophony(current_role_notes) # Pass a copy if needed
            print(f"    Note count after monophony: {len(current_role_notes)}")

        # Step 4: Velocity Quantization
        for note in current_role_notes:
            note["quantized_velocity"] = quantize_velocity(note["velocity"], num_velocity_bins)
        
        processed_notes_all_roles.extend(current_role_notes)

    # Sort all processed notes globally by start time, then pitch, then role (for stable order)
    # This global list of notes is what you'd feed into a Step 6 (Tokenization)
    processed_notes_all_roles.sort(key=lambda x: (x["start_tick"], x["pitch"], x["role"]))
    print(f"\nTotal processed notes for {midi_path}: {len(processed_notes_all_roles)}")
    # for note in processed_notes_all_roles[:10]: # Print a few samples
    #     print(f"    {note}")

    # Step 5
    tempos, key_signatures, time_signatures = extract_meta_info(mid)

    # At this point, you have:
    # - processed_notes_all_roles: A list of note dicts, each with pitch, start_tick, end_tick,
    #                              original_velocity, quantized_velocity, and role.
    # - tempos, key_signatures, time_signatures: Lists of meta events.
    #
    # The next step (not covered here) would be to convert this structured data
    # into a single sequence of tokens for your Transformer. This would involve
    # creating "Note_On_Role_Pitch", "Duration", "Velocity" tokens, and potentially
    # "Tempo", "Key", "TimeSignature" tokens, all ordered correctly in time.

    return {
        "file_path": midi_path,
        "ticks_per_beat": mid.ticks_per_beat,
        "processed_notes": processed_notes_all_roles,
        "tempos": tempos,
        "key_signatures": key_signatures,
        "time_signatures": time_signatures
    }


# %%
test = preprocess_midi_file(file_path[259])
test

# %%
df = pd.DataFrame(test['processed_notes'])
df

# %%
df['role'].unique()

# %%
from tqdm.notebook import tqdm
tracks = set()
tpb = []
for i in tqdm(file_path):
    test = preprocess_midi_file(file_path[259])
    df = pd.DataFrame(test['processed_notes'])
    tracks = tracks.union(df['role'].unique())
    tpb.append(test['ticks_per_beat'])

# %%
tracks

# %%
import numpy as np
np.unique([tpb])

# %%
DURATION_BINS_BEATS = sorted(list(set([
    0.0, 0.0625, 0.125, 0.1875, 0.25, 0.333, 0.375, 0.5, 
    0.666, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0
])))

TIME_SHIFT_BINS_BEATS = sorted(list(set([
    0.0, 0.0625, 0.125, 0.25, 0.333, 0.5, 0.666, 0.75, 
    1.0, 1.5, 2.0, 3.0, 4.0
])))

# Example: Quantize tempo into N bins (microseconds per beat)
TEMPO_BINS_US_PER_BEAT = [500000.0] # Corresponds to 240, 150, 120, 100, 80, 60 BPM

# Helper for quantization
def find_closest_bin_index(value, bins):
    """Finds the index of the bin boundary that the value is closest to (or falls into).
    For this simple version, it finds the largest bin boundary <= value.
    A more robust approach might use np.digitize or define ranges.
    """
    if not bins: return 0
    # Find where the value would be inserted to maintain order
    # Then take the index of the element to its left (meaning it's greater than or equal to that bin start)
    idx = np.searchsorted(bins, value, side='right') -1
    return max(0, idx) # Ensure non-negative index


# --- Tokenization Function ---
def tokenize_processed_data(processed_data, 
                            duration_bins_beats=DURATION_BINS_BEATS,
                            time_shift_bins_beats=TIME_SHIFT_BINS_BEATS,
                            tempo_bins_us=TEMPO_BINS_US_PER_BEAT,
                            include_tempo=True,
                            include_key_sig=True, # Basic key name tokenization
                            include_time_sig=True # Basic N/D tokenization
                           ):
    """
    Converts the structured processed_data into a sequence of string tokens.
    """
    ticks_per_beat = processed_data["ticks_per_beat"]
    if ticks_per_beat <= 0: # Safety check for very high/unusual ticks_per_beat
        print(f"Warning: Invalid ticks_per_beat: {ticks_per_beat}. Defaulting to 480 for quantization calculations.")
        # This is a fallback, ideally you'd understand why it's so high.
        # If it's consistently e.g. 22050 and represents something like ticks/second,
        # then beat-relative quantization here needs tempo to be accurate.
        # For now, this example proceeds assuming it can be used.
        # If tempo is fixed, ticks_per_beat might be ticks_per_second / (BPM/60)

    all_events = [] # List of (tick, event_type, event_value_or_sub_dict)

    # 1. Convert Notes to Event Tuples
    for note in processed_data["processed_notes"]:
        # A note event will be a sequence: Role, Pitch, Velocity, Duration
        # All are considered to start at note['start_tick'] for sorting purposes
        # but will be emitted sequentially in the token stream.
        duration_ticks = note["end_tick"] - note["start_tick"]
        duration_beats = duration_ticks / ticks_per_beat
        quantized_duration_bin_idx = find_closest_bin_index(duration_beats, duration_bins_beats)

        # We'll add them with sub-priorities to ensure order after sorting by tick
        all_events.append((note["start_tick"], 0, "role", note["role"]))
        all_events.append((note["start_tick"], 1, "pitch", note["pitch"]))
        all_events.append((note["start_tick"], 2, "quantized_velocity", note["quantized_velocity"]))
        all_events.append((note["start_tick"], 3, "duration", f"D{quantized_duration_bin_idx}")) # D0, D1 etc.

    # 2. Add Meta Events (if configured)
    if include_tempo:
        for tempo_event in processed_data["tempos"]:
            quantized_tempo_bin_idx = find_closest_bin_index(tempo_event["tempo_us_per_beat"], tempo_bins_us)
            all_events.append((tempo_event["tick"], -3, "tempo", f"T{quantized_tempo_bin_idx}")) # Negative prio for meta
    
    if include_key_sig:
        for key_sig_event in processed_data["key_signatures"]:
            # Basic token: just the key name string. More complex mapping could be done.
            all_events.append((key_sig_event["tick"], -2, "key_signature", str(key_sig_event["key"])))

    if include_time_sig:
        for time_sig_event in processed_data["time_signatures"]:
            ts_token = f"{time_sig_event['numerator']}_{time_sig_event['denominator']}"
            all_events.append((time_sig_event["tick"], -1, "time_signature", ts_token))

    # 3. Sort all_events: primary by tick, secondary by our defined sub-priority
    all_events.sort(key=lambda x: (x[0], x[1])) # x[0] is tick, x[1] is sub-priority

    # 4. Generate Final Token Stream (using Time Shift)
    token_sequence = ["<SOS>"]
    last_event_tick = 0

    for tick, sub_prio, event_type, event_value in all_events:
        # Calculate and add Time_Shift token
        time_shift_ticks = tick - last_event_tick
        if time_shift_ticks > 0:
            time_shift_beats = time_shift_ticks / ticks_per_beat
            quantized_timeshift_bin_idx = find_closest_bin_index(time_shift_beats, time_shift_bins_beats)
            token_sequence.append(f"Time_Shift_TS{quantized_timeshift_bin_idx}")
        
        # Add the actual event token
        if event_type == "role":
            token_sequence.append(f"Role_{event_value}")
        elif event_type == "pitch":
            token_sequence.append(f"Pitch_{event_value}")
        elif event_type == "quantized_velocity":
            token_sequence.append(f"Velocity_{event_value}") # Already quantized
        elif event_type == "duration":
            token_sequence.append(f"Duration_{event_value}") # Already binned string "D_idx"
        elif event_type == "tempo":
            token_sequence.append(f"Tempo_{event_value}") # Already binned string "T_idx"
        elif event_type == "key_signature":
            token_sequence.append(f"Key_{event_value.replace(' ', '_')}") # Sanitize key name
        elif event_type == "time_signature":
            token_sequence.append(f"TimeSig_{event_value}")
            
        last_event_tick = tick

    token_sequence.append("<EOS>")
    return token_sequence

# --- Vocabulary Building (Illustrative - run after tokenizing many files) ---
def build_vocabulary(all_token_sequences):
    """
    Builds a vocabulary (token to int mapping and vice-versa)
    from a list of token sequences.
    """
    all_tokens = set()
    for seq in all_token_sequences:
        for token in seq:
            all_tokens.add(token)
    
    # Add padding token if not present, though it's usually handled separately
    # all_tokens.add("<PAD>")


    sorted_tokens = sorted(list(all_tokens))
    token_to_int = {token: i for i, token in enumerate(sorted_tokens)}
    int_to_token = {i: token for token, i in token_to_int.items()}
    
    # Ensure <PAD> is 0 if you add it this way, or handle it explicitly
    # if "<PAD>" in token_to_int and token_to_int["<PAD>"] != 0:
    #     old_pad_idx = token_to_int["<PAD>"]
    #     token_at_zero = int_to_token[0]
    #     token_to_int["<PAD>"] = 0
    #     token_to_int[token_at_zero] = old_pad_idx
    #     int_to_token[0] = "<PAD>"
    #     int_to_token[old_pad_idx] = token_at_zero


    return token_to_int, int_to_token


# %%
token_sequence = tokenize_processed_data(
        preprocess_midi_file(file_path[369]),
        include_tempo=True,
        include_key_sig=True,
        include_time_sig=True
    )

# %%
token_sequence

# %%
def collect_all_duration_data(list_of_midi_files):
    all_durations_beats = []
    skipped_files = 0

    for midi_path in tqdm(list_of_midi_files): # tqdm for progress bar
        # --- Step 1: Parse ---
        parsed_output = parse_midi_basic_info(midi_path)
        if parsed_output is None:
            skipped_files += 1
            continue
        mid, track_info_list = parsed_output

        if mid.ticks_per_beat <= 0:
            print(f"Warning: Invalid ticks_per_beat ({mid.ticks_per_beat}) for {midi_path}. Skipping.")
            skipped_files += 1
            continue

        # --- Step 2: Identify Roles (Your custom logic is key here) ---
        track_roles = identify_track_roles(track_info_list, midi_path)

        # --- Utility: Extract notes ---
        notes_by_role_raw = extract_notes_from_tracks(mid, track_roles)

        # --- Step 3: Monophony (applied per role) & Duration Calculation ---
        roles_to_make_monophonic = ["LEAD_PULSE1", "HARMONY_PULSE2", "BASS"] # Customize!

        for role, notes in notes_by_role_raw.items():
            if not notes:
                continue

            current_role_notes = notes
            if role in roles_to_make_monophonic:
                current_role_notes = enforce_monophony(current_role_notes)

            for note in current_role_notes:
                duration_ticks = note["end_tick"] - note["start_tick"]
                if duration_ticks > 0 and mid.ticks_per_beat > 0:
                    duration_beats_val = duration_ticks / mid.ticks_per_beat
                    all_durations_beats.append(duration_beats_val)

    print(f"Processed {len(list_of_midi_files) - skipped_files} files. Skipped {skipped_files} files.")
    return all_durations_beats

durations = collect_all_duration_data(file_path)

# %%
import matplotlib.pyplot as plt

plt.hist(durations, bins=20, range=[0, 2])

# %%
def collect_all_timeshift_data(list_of_midi_files,
                             include_tempo=True, # Same flags as tokenize_processed_data
                             include_key_sig=True,
                             include_time_sig=True):
    all_time_shifts_beats = []
    skipped_files = 0

    for midi_path in tqdm(list_of_midi_files): # tqdm for progress bar
        # --- Use your full preprocess_midi_file to get structured data ---
        # This assumes preprocess_midi_file is defined and includes steps 1-5
        # You will need to define all your functions before this one.
        # To avoid re-defining, ensure your main script structure allows calling them.
        processed_data = preprocess_midi_file(midi_path, num_velocity_bins=8) # Use your chosen num_bins

        if processed_data is None:
            skipped_files += 1
            continue

        ticks_per_beat = processed_data["ticks_per_beat"]
        if ticks_per_beat <= 0:
            print(f"Warning: Invalid ticks_per_beat ({ticks_per_beat}) for {midi_path} in timeshift. Skipping.")
            skipped_files += 1
            continue

        # --- Reconstruct event list similar to tokenize_processed_data ---
        current_file_events = []
        # 1. Convert Notes
        for note in processed_data["processed_notes"]:
            # For time shift, we only care about the start_tick of the event group
            # The sub-priority ensures they are processed in order if at same tick
            current_file_events.append((note["start_tick"], 0, "role", note["role"]))
            current_file_events.append((note["start_tick"], 1, "pitch", note["pitch"]))
            current_file_events.append((note["start_tick"], 2, "quantized_velocity", note["quantized_velocity"]))
            # Duration event itself doesn't introduce a new time shift from its components
            # but its existence as part of the note "block" is important.
            # For time shift analysis, we just consider the start of this block.
            # To be precise, let's consider each token as a potential event point
            # This means a note will be: TimeShift -> Role -> TimeShift(0) -> Pitch -> TimeShift(0) -> Velocity -> TimeShift(0) -> Duration
            # For simplicity in THIS analysis, let's consider the start of each note "group" and meta events.

        # Simplified event list for time shift analysis:
        # Take the start_tick of each note, and ticks of meta-events
        simplified_event_ticks = []
        for note in processed_data["processed_notes"]:
            simplified_event_ticks.append(note["start_tick"])
        if include_tempo:
            for tempo_event in processed_data["tempos"]:
                simplified_event_ticks.append(tempo_event["tick"])
        if include_key_sig:
            for key_sig_event in processed_data["key_signatures"]:
                simplified_event_ticks.append(key_sig_event["tick"])
        if include_time_sig:
            for time_sig_event in processed_data["time_signatures"]:
                simplified_event_ticks.append(time_sig_event["tick"])

        if not simplified_event_ticks:
            continue

        # Remove duplicates and sort unique ticks
        sorted_unique_ticks = sorted(list(set(simplified_event_ticks)))

        if not sorted_unique_ticks:
            continue

        last_event_tick = 0
        # Add initial shift from 0 if first event is not at 0
        if sorted_unique_ticks[0] > 0:
             time_shift_ticks = sorted_unique_ticks[0] - 0
             all_time_shifts_beats.append(time_shift_ticks / ticks_per_beat)
             last_event_tick = sorted_unique_ticks[0]


        for i in range(len(sorted_unique_ticks)):
            current_event_tick = sorted_unique_ticks[i]
            time_shift_ticks = current_event_tick - last_event_tick
            if time_shift_ticks >= 0: # Should always be true if sorted
                all_time_shifts_beats.append(time_shift_ticks / ticks_per_beat)
            last_event_tick = current_event_tick

    print(f"Processed {len(list_of_midi_files) - skipped_files} files for time shifts. Skipped {skipped_files} files.")
    return all_time_shifts_beats
shift = collect_all_timeshift_data(file_path)

# %%
plt.hist(shift, bins=20, range=[0, 2])

# %%
def collect_all_tempo_data(list_of_midi_files):
    all_tempos_us_per_beat = []
    skipped_files = 0

    for midi_path in tqdm(list_of_midi_files): # tqdm for progress bar
        parsed_output = parse_midi_basic_info(midi_path) # Only need basic parse for meta
        if parsed_output is None:
            skipped_files += 1
            continue
        mid, _ = parsed_output

        # Use your existing extract_meta_info
        tempos, _, _ = extract_meta_info(mid) # Corrected function call
        for tempo_event in tempos:
            all_tempos_us_per_beat.append(tempo_event["tempo_us_per_beat"])

    # Add default tempo if some files had no explicit tempo
    # (Mido default is 500000 if no set_tempo event is found,
    # but extract_meta_info only reports found events. You might need to
    # add a default for files that return an empty tempos list from extract_meta_info)
    # For simplicity, we'll analyze only explicitly set tempos here.
    # If many files have no tempo events, you might want to add 500000 to the list for each such file.

    print(f"Processed {len(list_of_midi_files) - skipped_files} files for tempos. Skipped {skipped_files} files.")
    return all_tempos_us_per_beat
tempo = collect_all_tempo_data(file_path)

# %%
plt.hist(tempo, bins=20, )

# %%
tempo

# %%
DURATION_BINS_BEATS = sorted(list(set([
    0.0, 0.0625, 0.125, 0.1875, 0.25, 0.333, 0.375, 0.5,
    0.666, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0
])))

TIME_SHIFT_BINS_BEATS = sorted(list(set([
    0.0, 0.0625, 0.125, 0.25, 0.333, 0.5, 0.666, 0.75,
    1.0, 1.5, 2.0, 3.0, 4.0
])))

# Assuming tempo is consistently ~120 BPM (500,000 µs/beat)
TEMPO_BINS_US_PER_BEAT = [490000.0] # Values >= 490000 will be Tempo_T0

# Tokenizer settings
TOKENIZER_SETTINGS = {
    "duration_bins_beats": DURATION_BINS_BEATS,
    "time_shift_bins_beats": TIME_SHIFT_BINS_BEATS,
    "tempo_bins_us": TEMPO_BINS_US_PER_BEAT,
    "include_tempo": True,  # Set to False if you decide not to tokenize fixed tempo
    "include_key_sig": True, # Or False
    "include_time_sig": True # Or False
}

# Preprocessing settings
NUM_VELOCITY_BINS = 8 # Example, choose based on your 8-bit style needs

list_of_your_midi_paths = file_path

# --- 2. Preprocess all MIDI Files ---
# This loop calls Steps 1-5 (as bundled in preprocess_midi_file)
all_processed_data = []
print(f"\n--- Starting MIDI Preprocessing for {len(list_of_your_midi_paths)} files ---")
for midi_path in tqdm(list_of_your_midi_paths, desc="Preprocessing MIDIs"):
    # Ensure your identify_track_roles function is well-customized before this step!
    processed_data = preprocess_midi_file(midi_path, num_velocity_bins=NUM_VELOCITY_BINS)
    if processed_data:
        all_processed_data.append(processed_data)
        # Optional: Save intermediate processed data
        # with open(f"{os.path.splitext(midi_path)[0]}_processed.json", "w") as f:
        #     json.dump(processed_data, f, indent=2)

print(f"\n--- Completed MIDI Preprocessing. {len(all_processed_data)} files processed successfully. ---")

# --- 3. Tokenize all Preprocessed Data ---
# This loop calls the tokenization function (Step 6)
all_token_sequences_str = []
print(f"\n--- Starting Tokenization for {len(all_processed_data)} processed files ---")
for processed_data_item in tqdm(all_processed_data, desc="Tokenizing Data"):
    token_sequence = tokenize_processed_data(
        processed_data_item,
        duration_bins_beats=TOKENIZER_SETTINGS["duration_bins_beats"],
        time_shift_bins_beats=TOKENIZER_SETTINGS["time_shift_bins_beats"],
        tempo_bins_us=TOKENIZER_SETTINGS["tempo_bins_us"],
        include_tempo=TOKENIZER_SETTINGS["include_tempo"],
        include_key_sig=TOKENIZER_SETTINGS["include_key_sig"],
        include_time_sig=TOKENIZER_SETTINGS["include_time_sig"]
    )
    all_token_sequences_str.append(token_sequence)

print(f"\n--- Completed Tokenization. {len(all_token_sequences_str)} sequences generated. ---")

if all_token_sequences_str:
    print(f"Example token sequence (first 20 tokens from first file): {all_token_sequences_str[0][:20]}")

# --- 4. Build Vocabulary ---
print("\n--- Building Vocabulary ---")
if not all_token_sequences_str:
    print("No token sequences to build vocabulary from. Exiting.")
else:
    token_to_int, int_to_token = build_vocabulary(all_token_sequences_str)
    vocab_size = len(token_to_int)
    print(f"Vocabulary Size: {vocab_size}")
    # Optional: Save vocabulary
    # with open("vocabulary.json", "w") as f:
    #     json.dump({"token_to_int": token_to_int, "int_to_token": int_to_token}, f, indent=2)
    # print("First 20 vocabulary items:")
    # for i, (token, index) in enumerate(token_to_int.items()):
    #     if i >= 20: break
    #     print(f"  '{token}': {index}")

    # --- 5. Convert Token Sequences to Integer Sequences ---
    all_token_sequences_int = []
    print("\n--- Converting to Integer Sequences ---")
    for token_seq_str in tqdm(all_token_sequences_str, desc="Integer Encoding"):
        int_sequence = [token_to_int[token] for token in token_seq_str]
        all_token_sequences_int.append(int_sequence)
    
    print(f"\n--- Completed Integer Encoding. {len(all_token_sequences_int)} sequences converted. ---")
    if all_token_sequences_int:
        print(f"Example integer sequence (first 20 tokens from first file): {all_token_sequences_int[0][:20]}")


# %%
token_to_int

# %%
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"
PAD_TOKEN = "<PAD>"
if PAD_TOKEN not in token_to_int:
    print("yes")

# %%
import numpy as np # If not already imported

# --- Assume this is your current vocabulary loaded from your data ---
# This is based on the image you provided (image_59c955.png)
# In your actual code, you would load or use your existing token_to_int
# For this example, I'll create a small version resembling your image:
current_token_to_int = {'<EOS>': 0,
 '<SOS>': 1,
 'Duration_D0': 2,
 'Duration_D1': 3,
 'Duration_D10': 4,
 'Duration_D11': 5,
 'Duration_D12': 6,
 'Duration_D13': 7,
 'Duration_D14': 8,
 'Duration_D15': 9,
 'Duration_D2': 10,
 'Duration_D3': 11,
 'Duration_D4': 12,
 'Duration_D5': 13,
 'Duration_D6': 14,
 'Duration_D7': 15,
 'Duration_D8': 16,
 'Duration_D9': 17,
 'Pitch_100': 18,
 'Pitch_101': 19,
 'Pitch_102': 20,
 'Pitch_103': 21,
 'Pitch_104': 22,
 'Pitch_105': 23,
 'Pitch_106': 24,
 'Pitch_107': 25,
 'Pitch_108': 26,
 'Pitch_21': 27,
 'Pitch_22': 28,
 'Pitch_23': 29,
 'Pitch_24': 30,
 'Pitch_25': 31,
 'Pitch_26': 32,
 'Pitch_27': 33,
 'Pitch_28': 34,
 'Pitch_29': 35,
 'Pitch_30': 36,
 'Pitch_31': 37,
 'Pitch_32': 38,
 'Pitch_33': 39,
 'Pitch_34': 40,
 'Pitch_35': 41,
 'Pitch_36': 42,
 'Pitch_37': 43,
 'Pitch_38': 44,
 'Pitch_39': 45,
 'Pitch_40': 46,
 'Pitch_41': 47,
 'Pitch_42': 48,
 'Pitch_43': 49,
 'Pitch_44': 50,
 'Pitch_45': 51,
 'Pitch_46': 52,
 'Pitch_47': 53,
 'Pitch_48': 54,
 'Pitch_49': 55,
 'Pitch_50': 56,
 'Pitch_51': 57,
 'Pitch_52': 58,
 'Pitch_53': 59,
 'Pitch_54': 60,
 'Pitch_55': 61,
 'Pitch_56': 62,
 'Pitch_57': 63,
 'Pitch_58': 64,
 'Pitch_59': 65,
 'Pitch_60': 66,
 'Pitch_61': 67,
 'Pitch_62': 68,
 'Pitch_63': 69,
 'Pitch_64': 70,
 'Pitch_65': 71,
 'Pitch_66': 72,
 'Pitch_67': 73,
 'Pitch_68': 74,
 'Pitch_69': 75,
 'Pitch_70': 76,
 'Pitch_71': 77,
 'Pitch_72': 78,
 'Pitch_73': 79,
 'Pitch_74': 80,
 'Pitch_75': 81,
 'Pitch_76': 82,
 'Pitch_77': 83,
 'Pitch_78': 84,
 'Pitch_79': 85,
 'Pitch_80': 86,
 'Pitch_81': 87,
 'Pitch_82': 88,
 'Pitch_83': 89,
 'Pitch_84': 90,
 'Pitch_85': 91,
 'Pitch_86': 92,
 'Pitch_87': 93,
 'Pitch_88': 94,
 'Pitch_89': 95,
 'Pitch_90': 96,
 'Pitch_91': 97,
 'Pitch_92': 98,
 'Pitch_93': 99,
 'Pitch_94': 100,
 'Pitch_95': 101,
 'Pitch_96': 102,
 'Pitch_97': 103,
 'Pitch_98': 104,
 'Pitch_99': 105,
 'Role_BASS': 106,
 'Role_LEAD_PULSE1': 107,
 'Tempo_T0': 108,
 'TimeSig_1_1': 109,
 'TimeSig_4_4': 110,
 'Time_Shift_TS0': 111,
 'Time_Shift_TS1': 112,
 'Time_Shift_TS10': 113,
 'Time_Shift_TS11': 114,
 'Time_Shift_TS12': 115,
 'Time_Shift_TS2': 116,
 'Time_Shift_TS3': 117,
 'Time_Shift_TS4': 118,
 'Time_Shift_TS5': 119,
 'Time_Shift_TS6': 120,
 'Time_Shift_TS7': 121,
 'Time_Shift_TS8': 122,
 'Time_Shift_TS9': 123,
 'Velocity_0': 124}
# --- End of example current_token_to_int ---

print(f"Old vocabulary size: {len(current_token_to_int)}")
print(f"Old <EOS> ID: {current_token_to_int.get(EOS_TOKEN)}") # Should be 0 from your image
print(f"Old <SOS> ID: {current_token_to_int.get(SOS_TOKEN)}") # Should be 1 from your image

# Extract all tokens other than the current <SOS> and <EOS>
# (as their string names are now fixed by our constants above)
other_tokens = []
for token_str, token_id in current_token_to_int.items():
    if token_str != EOS_TOKEN and token_str != SOS_TOKEN:
        other_tokens.append(token_str)

# Sort these other tokens alphabetically for a consistent order
sorted_other_tokens = sorted(list(set(other_tokens))) # Use set to ensure uniqueness if any issues

# Create the new vocabulary list with PAD, SOS, EOS at the beginning
new_vocab_list = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN] + sorted_other_tokens

# Create new mappings
token_to_int_new = {token: i for i, token in enumerate(new_vocab_list)}
int_to_token_new = {i: token for token, i in enumerate(new_vocab_list)}

# --- Update your global vocabulary variables ---
token_to_int = token_to_int_new
int_to_token = int_to_token_new
vocab_size = len(token_to_int)

pad_idx = token_to_int[PAD_TOKEN]
sos_idx = token_to_int[SOS_TOKEN]
eos_idx = token_to_int[EOS_TOKEN]

print("\n--- Updated Vocabulary ---")
print(f"New vocabulary size: {vocab_size}") # Should be old_size + 1 (if <PAD> was truly new)
                                         # or old_size if <SOS>/<EOS> were just re-mapped
print(f"PAD_TOKEN ('{PAD_TOKEN}') ID: {pad_idx}")
print(f"SOS_TOKEN ('{SOS_TOKEN}') ID: {sos_idx}")
print(f"EOS_TOKEN ('{EOS_TOKEN}') ID: {eos_idx}")

# Verify a few other tokens to see their new IDs
if 'Duration_D0' in token_to_int:
    print(f"New 'Duration_D0' ID: {token_to_int['Duration_D0']}") # Will be shifted by 1 or 2 compared to old
if 'Pitch_100' in token_to_int:
    print(f"New 'Pitch_100' ID: {token_to_int['Pitch_100']}")

# It's important that your integer sequences (all_token_sequences_int)
# are re-generated using this NEW token_to_int map if they were already created
# using the old one. If you only have all_token_sequences_STR, you're fine,
# as you'll convert them to integers using this new map.

# %%
token_to_int

# %%
all_token_sequences_str

# %%
all_token_sequences_int

# %%
int_to_token

# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, Subset
from torch.nn.utils.rnn import pad_sequence
import math
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold
import os

# --- [Your existing MusicDataset, Collate Function, PositionalEncoding, MusicTransformer classes go here] ---
# (Make sure these are defined exactly as in the previous combined script)
# --- For brevity, I'm omitting them here, but they MUST be in your actual script ---

# --- Assume these are defined from your previous steps ---
# token_to_int: Your full vocabulary e.g. {'<PAD>': 0, '<SOS>': 1, ...}
# all_token_sequences_str: List of lists of string tokens from all your files

# --- 0. Ensure Vocab and Convert String Sequences to Integer Sequences ---
if 'token_to_int' not in globals() or not token_to_int:
    # Create dummy token_to_int if it's not defined (for this standalone example to run)
    print("Warning: token_to_int not defined. Using dummy vocab.")
    token_to_int = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, 'Role_A':3, 'Pitch_60':4, 'Vel_0':5, 'Dur_D0':6, 'TS_0':7}
    # In your real script, this should already be properly loaded
if '<PAD>' not in token_to_int or token_to_int['<PAD>'] != 0:
    raise ValueError("'<PAD>' token must be in token_to_int and have an ID of 0.")

if 'all_token_sequences_str' not in globals() or not all_token_sequences_str:
    print("Warning: all_token_sequences_str is empty. Using dummy data for demonstration.")
    all_token_sequences_str = [
         ['<SOS>', 'Role_A', 'Pitch_60', 'Vel_0', 'Dur_D0', 'TS_0', 'Role_A', 'Pitch_60', '<EOS>'],
         ['<SOS>', 'Role_A', 'Pitch_60', 'Vel_0', 'Dur_D0', '<EOS>']
     ] * 100 # Make a bit more data for CV example

all_token_sequences_int = []
for str_seq in all_token_sequences_str:
    all_token_sequences_int.append([token_to_int.get(token, token_to_int['<PAD>']) for token in str_seq])

pad_idx = token_to_int["<PAD>"]
sos_idx = token_to_int["<SOS>"] # Not directly used in training loop but good to have
eos_idx = token_to_int["<EOS>"] # Not directly used in training loop but good to have
vocab_size = len(token_to_int)

# --- 1. Configuration for Data Preparation & Model ---
MAX_SEQUENCE_LENGTH = 32
BATCH_SIZE = 16
NUM_EPOCHS = 50 # Epochs PER FOLD
K_FOLDS = 10
LEARNING_RATE = 0.0005
D_MODEL = 64 # Smaller for faster demo
NHEAD = 2
NUM_DECODER_LAYERS = 2
DIM_FEEDFORWARD = 128
DROPOUT = 0.1

# --- Device Setup ---
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# --- MusicDataset Definition ---
class MusicDataset(Dataset):
    def __init__(self, integer_sequences, max_model_input_len):
        self.sequences = []
        for seq_tensor in integer_sequences:
            if len(seq_tensor) < 2: continue
            self.sequences.append(seq_tensor[:max_model_input_len + 1])
    def __len__(self): return len(self.sequences)
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        x = seq[:-1]
        y = seq[1:]
        return x, y

# --- Collate Function Definition ---
def collate_fn(batch, pad_idx, max_len):
    inputs, targets = zip(*batch)
    inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=pad_idx)
    if inputs_padded.size(1) < max_len:
        padding = torch.full((inputs_padded.size(0), max_len - inputs_padded.size(1)),
                             pad_idx, dtype=torch.long, device=inputs_padded.device)
        inputs_padded = torch.cat([inputs_padded, padding], dim=1)
    elif inputs_padded.size(1) > max_len:
        inputs_padded = inputs_padded[:, :max_len]

    targets_padded = pad_sequence(targets, batch_first=True, padding_value=pad_idx)
    if targets_padded.size(1) < max_len:
        padding = torch.full((targets_padded.size(0), max_len - targets_padded.size(1)),
                             pad_idx, dtype=torch.long, device=targets_padded.device)
        targets_padded = torch.cat([targets_padded, padding], dim=1)
    elif targets_padded.size(1) > max_len:
        targets_padded = targets_padded[:, :max_len]
    return inputs_padded, targets_padded

# --- Positional Encoding Definition ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

# --- MusicTransformer Definition ---
class MusicTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_decoder_layers, dim_feedforward, dropout=0.1, max_seq_len=MAX_SEQUENCE_LENGTH):
        super(MusicTransformer, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=max_seq_len)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, activation='gelu'
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def _generate_square_subsequent_mask(self, sz, device):
        mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, src_padding_mask=None):
        seq_len = src.size(1)
        tgt_mask = self._generate_square_subsequent_mask(seq_len, src.device)
        embedded_src = self.embedding(src) * math.sqrt(self.d_model)
        pos_encoded_src = self.pos_encoder(embedded_src)
        output = self.transformer_decoder(
            tgt=pos_encoded_src, memory=pos_encoded_src, tgt_mask=tgt_mask, memory_mask=None,
            tgt_key_padding_mask=src_padding_mask, memory_key_padding_mask=src_padding_mask
        )
        return self.fc_out(output)

# --- Helper Function: Train for one epoch ---
def train_epoch(model, dataloader, optimizer, criterion, device, current_vocab_size, current_pad_idx):
    model.train()
    epoch_loss = 0
    for x_batch, y_batch in tqdm(dataloader, desc="Training", leave=False, ncols=100):
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        src_padding_mask = (x_batch == current_pad_idx)
        output_logits = model(x_batch, src_padding_mask=src_padding_mask)
        loss = criterion(output_logits.view(-1, current_vocab_size), y_batch.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(dataloader)

# --- Helper Function: Evaluate the model ---
def evaluate(model, dataloader, criterion, device, current_vocab_size, current_pad_idx):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for x_batch, y_batch in tqdm(dataloader, desc="Evaluating", leave=False, ncols=100):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            src_padding_mask = (x_batch == current_pad_idx)
            output_logits = model(x_batch, src_padding_mask=src_padding_mask)
            loss = criterion(output_logits.view(-1, current_vocab_size), y_batch.view(-1))
            epoch_loss += loss.item()
    return epoch_loss / len(dataloader)

# --- Main Cross-Validation Training Loop ---
print(f"\n--- Starting {K_FOLDS}-Fold Cross-Validation ---")

tensor_sequences = [torch.tensor(seq, dtype=torch.long) for seq in all_token_sequences_int]
full_music_dataset = MusicDataset(tensor_sequences, MAX_SEQUENCE_LENGTH)

kfold = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)

# Lists to store overall results from each fold
overall_best_val_losses_per_fold = []

# Lists to store epoch-wise history for ALL folds (for plotting)
all_folds_train_loss_history = []
all_folds_val_loss_history = []


custom_collate_fn = lambda b: collate_fn(b, pad_idx, MAX_SEQUENCE_LENGTH)

for fold, (train_ids, val_ids) in enumerate(kfold.split(full_music_dataset)):
    print(f"\n===== FOLD {fold + 1}/{K_FOLDS} =====")

    train_subsampler = SubsetRandomSampler(train_ids)
    val_subsampler = SubsetRandomSampler(val_ids)

    current_train_dataloader = DataLoader(full_music_dataset, batch_size=BATCH_SIZE,
                                          sampler=train_subsampler, collate_fn=custom_collate_fn)
    current_val_dataloader = DataLoader(full_music_dataset, batch_size=BATCH_SIZE,
                                        sampler=val_subsampler, collate_fn=custom_collate_fn)

    model = MusicTransformer(
        vocab_size=vocab_size, d_model=D_MODEL, nhead=NHEAD,
        num_decoder_layers=NUM_DECODER_LAYERS, dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT, max_seq_len=MAX_SEQUENCE_LENGTH
    ).to(device)
    
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.7)

    print(f"Fold {fold+1} - Train size: {len(train_ids)}, Val size: {len(val_ids)}")

    fold_best_val_loss = float('inf')
    current_fold_train_loss_epochs = [] # Store epoch losses for THIS fold
    current_fold_val_loss_epochs = []   # Store epoch losses for THIS fold

    for epoch in range(1, NUM_EPOCHS + 1):
        # print(f"--- Fold {fold + 1}, Epoch {epoch}/{NUM_EPOCHS} ---") # Less verbose
        train_loss = train_epoch(model, current_train_dataloader, optimizer, criterion, device, vocab_size, pad_idx)
        val_loss = evaluate(model, current_val_dataloader, criterion, device, vocab_size, pad_idx)
        
        current_fold_train_loss_epochs.append(train_loss)
        current_fold_val_loss_epochs.append(val_loss)

        if scheduler:
            scheduler.step()

        print(f"\tFold {fold+1} Epoch {epoch} - Train Loss: {train_loss:.4f} (PPL: {math.exp(train_loss):.2f}) | "
              f"Val Loss: {val_loss:.4f} (PPL: {math.exp(val_loss):.2f}) | "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")

        if val_loss < fold_best_val_loss:
            fold_best_val_loss = val_loss
            # torch.save(model.state_dict(), f"music_transformer_fold{fold+1}_best_epoch{epoch}.pth")
            # print(f"\tNew best val_loss for fold {fold+1}: {fold_best_val_loss:.4f}")
    
    overall_best_val_losses_per_fold.append(fold_best_val_loss)
    all_folds_train_loss_history.append(current_fold_train_loss_epochs) # Append list of epoch losses
    all_folds_val_loss_history.append(current_fold_val_loss_epochs)   # Append list of epoch losses
    print(f"--- Finished Fold {fold+1}/{K_FOLDS} --- Best Val Loss for this fold: {fold_best_val_loss:.4f} ---")


print("\n--- Cross-Validation Finished ---")
if overall_best_val_losses_per_fold:
    mean_val_loss = np.mean(overall_best_val_losses_per_fold)
    std_val_loss = np.std(overall_best_val_losses_per_fold)
    print(f"Average Best Validation Loss across {K_FOLDS} folds: {mean_val_loss:.4f} +/- {std_val_loss:.4f}")
    print(f"Average Best Validation PPL across {K_FOLDS} folds: {math.exp(mean_val_loss):.2f}")
else:
    print("No validation results to average.")


# --- Plotting All Folds' Loss Curves ---
epochs_range = range(1, NUM_EPOCHS + 1)

# Plot Training Losses for all folds
plt.figure(figsize=(12, 7))
for i, fold_train_loss in enumerate(all_folds_train_loss_history):
    plt.plot(epochs_range, fold_train_loss, linestyle='--', alpha=0.7, label=f'Fold {i+1} Train Loss')
if all_folds_train_loss_history: # Calculate and plot mean training loss
    mean_train_losses = np.mean(all_folds_train_loss_history, axis=0)
    plt.plot(epochs_range, mean_train_losses, color='black', linewidth=2, label='Mean Train Loss')
plt.title('Training Loss Curves Across All Folds')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.grid(True)
plt.xticks(list(epochs_range))
plt.tight_layout()
plt.show()

# Plot Validation Losses for all folds
plt.figure(figsize=(12, 7))
for i, fold_val_loss in enumerate(all_folds_val_loss_history):
    plt.plot(epochs_range, fold_val_loss, linestyle='-', alpha=0.7, label=f'Fold {i+1} Val Loss')
if all_folds_val_loss_history: # Calculate and plot mean validation loss
    mean_val_losses = np.mean(all_folds_val_loss_history, axis=0)
    plt.plot(epochs_range, mean_val_losses, color='black', linewidth=2, label='Mean Val Loss')
plt.title('Validation Loss Curves Across All Folds')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.grid(True)
plt.xticks(list(epochs_range))
plt.tight_layout()
plt.show()

# %%
plt.figure(figsize=(12, 7))
for i, fold_train_loss in enumerate(all_folds_train_loss_history):
    plt.plot(epochs_range, fold_train_loss, linestyle='--', alpha=0.7, label=f'Fold {i+1} Train Loss')
if all_folds_train_loss_history: # Calculate and plot mean training loss
    mean_train_losses = np.mean(all_folds_train_loss_history, axis=0)
    plt.plot(epochs_range, mean_train_losses, color='black', linewidth=2, label='Mean Train Loss')
plt.title('Training Loss Curves Across All Folds')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.grid(True)
plt.xticks(list(epochs_range))
plt.tight_layout()
plt.show()

# Plot Validation Losses for all folds
plt.figure(figsize=(12, 7))
for i, fold_val_loss in enumerate(all_folds_val_loss_history):
    plt.plot(epochs_range, fold_val_loss, linestyle='-', alpha=0.7, label=f'Fold {i+1} Val Loss')
if all_folds_val_loss_history: # Calculate and plot mean validation loss
    mean_val_losses = np.mean(all_folds_val_loss_history, axis=0)
    plt.plot(epochs_range, mean_val_losses, color='black', linewidth=2, label='Mean Val Loss')
plt.title('Validation Loss Curves Across All Folds')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.grid(True)
plt.xticks(list(epochs_range))
plt.tight_layout()
plt.show()

# %%
import torch
import torch.nn.functional as F
import math
import numpy as np # For np.searchsorted if you use it in find_closest_bin_index_from_token

# --- Assume these are defined globally or passed as arguments ---
# DURATION_BINS_BEATS = [...]
# TIME_SHIFT_BINS_BEATS = [...]
# TEMPO_BINS_US_PER_BEAT = [...] # Though for fixed tempo, this might not be used in generation if tempo isn't generated
# int_to_token = { ... }
# token_to_int = { ... }
# pad_idx = token_to_int["<PAD>"]
# sos_idx = token_to_int["<SOS>"]
# eos_idx = token_to_int["<EOS>"]
# vocab_size = len(token_to_int)
# device = torch.device(...)
# model = Your trained MusicTransformer model

# MAX_SEQUENCE_LENGTH (model_context_window): The sequence length the model was trained on.
# This is used to prepare input for the model.
# Example: MAX_SEQUENCE_LENGTH = 256

# Helper function to parse token value (assuming format like "Time_Shift_TS7")
def get_bin_index_from_token_value(token_value_str, prefix_to_remove):
    try:
        return int(token_value_str.replace(prefix_to_remove, ""))
    except ValueError:
        print(f"Warning: Could not parse bin index from {token_value_str} after removing {prefix_to_remove}")
        return None # Or a default error index

# Helper to get beat value from bin index (same as in detokenizer)
def get_value_from_bin_index(bin_idx, bins_list):
    if 0 <= bin_idx < len(bins_list):
        return bins_list[bin_idx]
    print(f"Warning: bin_idx {bin_idx} out of range for bins list of length {len(bins_list)}. Defaulting to first bin.")
    return bins_list[0] if bins_list else 0.0


def generate_sequence_for_duration(
    model,
    start_tokens_int,
    target_duration_seconds,
    ticks_per_beat,
    bpm, # Add BPM to calculate target ticks
    time_shift_bins_beats, # Pass the bin definitions
    max_output_tokens,     # Max number of tokens to generate in total
    temperature, top_k, top_p,
    device, current_eos_idx, current_pad_idx, model_context_window):

    model.eval()
    generated_sequence_int = start_tokens_int[:]
    
    # Calculate target duration in ticks
    beats_per_second = bpm / 60.0
    target_total_ticks = int(target_duration_seconds * beats_per_second * ticks_per_beat)
    
    current_accumulated_ticks = 0

    print(f"Targeting duration: {target_duration_seconds}s, which is {target_total_ticks} ticks at {bpm} BPM with {ticks_per_beat} tpb.")

    with torch.no_grad():
        for i in range(max_output_tokens - len(start_tokens_int)):
            # Prepare input for the model
            # Take up to model_context_window last tokens
            if len(generated_sequence_int) > model_context_window:
                input_seq_current = generated_sequence_int[-model_context_window:]
            else:
                input_seq_current = generated_sequence_int[:]
            
            input_tensor = torch.tensor([input_seq_current], dtype=torch.long).to(device)
            
            # For generation of a single sequence, padding mask is not strictly needed
            # if the input sequence is always <= model_context_window.
            # If input_seq_current could be shorter than model_context_window AND your model
            # was trained with padding, you might need to pad it here.
            # However, usually for generation, we feed the exact current sequence.
            output_logits = model(input_tensor, src_padding_mask=None)
            
            # Get logits for the very last token position
            next_token_logits = output_logits[0, -1, :] / temperature
            
            # Apply top-k filtering
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(next_token_logits, top_k)
                next_token_logits[next_token_logits < v[-1]] = -float('Inf')
            
            # Apply top-p (nucleus) filtering
            if top_p is not None and top_p > 0.0 and top_p < 1.0: # Ensure top_p is valid
                probabilities_for_nucleus = torch.softmax(next_token_logits, dim=-1)
                sorted_probs, sorted_indices = torch.sort(probabilities_for_nucleus, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[indices_to_remove] = -float('Inf')
                
            probabilities = F.softmax(next_token_logits, dim=-1)
            next_token_int = torch.multinomial(probabilities, 1).item()
            
            generated_sequence_int.append(next_token_int)
            
            # Update accumulated ticks based on the generated token
            token_str = int_to_token.get(next_token_int, "") # Get the string representation
            if token_str.startswith("Time_Shift_TS"):
                try:
                    bin_idx = int(token_str.split("TS")[-1])
                    time_shift_b = get_value_from_bin_index(bin_idx, time_shift_bins_beats)
                    current_accumulated_ticks += int(round(time_shift_b * ticks_per_beat))
                except ValueError:
                    print(f"Warning: Could not parse time shift token for duration tracking: {token_str}")

            if next_token_int == current_eos_idx:
                print("EOS token generated.")
                break
            
            if current_accumulated_ticks >= target_total_ticks:
                print(f"Target duration reached at {current_accumulated_ticks} ticks (approx {current_accumulated_ticks / (beats_per_second * ticks_per_beat):.2f}s).")
                # Optionally, try to end with an EOS or complete the current musical phrase gracefully
                # For simplicity here, we just break.
                break
        
        # If loop finished due to max_output_tokens but not EOS, append EOS if not already there.
        if generated_sequence_int[-1] != current_eos_idx:
            generated_sequence_int.append(current_eos_idx)
            
    return generated_sequence_int

# %%
TEMPERATURE = 0.99
TOP_K = 50
TOP_P = 0.0
SEQUENCE_LENGTH = 2000
TARGET_DURATION_SECONDS = 30
TICKS_PER_BEAT_DETOK = 22050

start_prompt_str = ["<SOS>"]
start_tokens_int = [token_to_int.get(s, pad_idx) for s in start_prompt_str]

generated_int_sequence_duration = generate_sequence_for_duration(
        model,
        start_tokens_int,
        target_duration_seconds=TARGET_DURATION_SECONDS,
        ticks_per_beat=TICKS_PER_BEAT_DETOK,
        bpm=60,
        time_shift_bins_beats=TIME_SHIFT_BINS_BEATS, # Pass the bins
        max_output_tokens=SEQUENCE_LENGTH,
        temperature=0.8, # Try a slightly higher temperature
        top_k=TOP_K,
        top_p=TOP_P,
        device=device,
        current_eos_idx=eos_idx,
        current_pad_idx=pad_idx,
        model_context_window=MAX_SEQUENCE_LENGTH # This is the model's trained input window size
    )
generated_int_sequence_duration

# %%
int_to_token = {i: t for t, i in token_to_int.items()}
generated_str_sequence = [int_to_token.get(i, "<UNK>") for i in generated_int_sequence]

print("--- Generated String Token Sequence ---")
for i in range(0, len(generated_str_sequence), 10): # Print 10 tokens per line
    print(" ".join(generated_str_sequence[i:i+10]))

# %%
import mido
from collections import defaultdict
import numpy as np # For np.array if needed for bins

# --- Assumed Configurations (these should match your tokenization) ---
# TICKS_PER_BEAT = 22050 # CRUCIAL: Set this to your dataset's value
# DURATION_BINS_BEATS = sorted(list(set([...]))) # As defined previously
# TIME_SHIFT_BINS_BEATS = sorted(list(set([...]))) # As defined previously
# TEMPO_BINS_US_PER_BEAT = [...] # As defined previously
# NUM_VELOCITY_BINS = 8 # Example, match your training config

# --- Helper: Map bin index back to a representative beat/value ---
def get_value_from_bin_index(bin_idx, bins_list):
    """
    Gets the representative value for a bin index.
    For simplicity, returns the lower bound of the bin.
    A more sophisticated approach might return the midpoint if bin edges were stored.
    """
    if 0 <= bin_idx < len(bins_list):
        return bins_list[bin_idx]
    # Handle out-of-bounds index, though ideally model output is constrained by vocab
    print(f"Warning: bin_idx {bin_idx} out of range for bins list of length {len(bins_list)}")
    return bins_list[0] # Default to the first bin's value

# --- Helper: Map quantized velocity bin back to MIDI velocity ---
def map_quantized_velocity_to_midi(quantized_bin_idx, num_total_velocity_bins):
    """
    Maps a quantized velocity bin index back to an approximate MIDI velocity (0-127).
    This example maps to the midpoint of the bin's range.
    """
    if num_total_velocity_bins <= 0: return 64 # Default
    bin_size_approx = 127.0 / num_total_velocity_bins
    # Calculate midpoint of the bin
    # Bin 0: (0*size + (0+1)*size)/2 = size/2
    # Bin V: (V*size + (V+1)*size)/2 = (2V+1)*size/2
    # (Ensure velocity is at least 1 for note_on, 0 is for note_off)
    velocity = int( (quantized_bin_idx + 0.5) * bin_size_approx )
    return max(1, min(127, velocity)) # Ensure it's a valid MIDI velocity > 0

# --- Helper: Define a mapping from Role Name to GM Program (optional) ---
# This helps MIDI players choose a somewhat appropriate sound.
# For 8-bit, the final sound often comes from a specific soundfont/emulator.
ROLE_TO_PROGRAM = {
    "LEAD_PULSE1": 80,  # GM Square Lead / Lead 1 (Square)
    "HARMONY_PULSE2": 80, # Could be different, e.g., Synth Pad
    "BASS": 33,         # GM Electric Bass (finger)
    "NOISE": 118,       # GM Synth Drum / Melodic Tom (for percussion-like sounds)
    "OTHER": 0          # GM Acoustic Grand Piano
}

def detokenize_to_midi(string_token_sequence,
                        ticks_per_beat,
                        duration_bins_beats,
                        time_shift_bins_beats,
                        tempo_bins_us,
                        num_velocity_bins,
                        output_midi_path="generated_music.mid"):
    """
    Converts a sequence of string tokens back into a MIDI file.
    """
    mid = mido.MidiFile(type=1, ticks_per_beat=ticks_per_beat)
    
    # Track for meta messages (tempo, time_sig, key_sig)
    meta_track = mido.MidiTrack()
    mid.tracks.append(meta_track)
    # Enforce at least one event on meta track if needed, or mido might drop it.
    # We'll add time signature and tempo if present.
    
    # Tracks for different roles
    role_tracks = {} # role_name -> MidiTrack
    role_last_abs_tick = defaultdict(int) # role_name -> last absolute tick time of an event on this track

    current_abs_tick = 0
    active_role = None
    pending_note_info = {} # Stores {'pitch': P, 'velocity_bin': V} for the active_role

    # Default meta messages if not specified early
    if not any(token.startswith("TimeSig_") for token in string_token_sequence):
        meta_track.append(mido.MetaMessage('time_signature', numerator=4, denominator=4, clocks_per_click=24, notated_32nd_notes_per_beat=8, time=0))
    if not any(token.startswith("Tempo_") for token in string_token_sequence):
         meta_track.append(mido.MetaMessage('set_tempo', tempo=500000, time=0)) # Default 120 BPM
    
    meta_track_last_abs_tick = 0

    for token_str in string_token_sequence:
        if token_str == "<SOS>":
            continue
        if token_str == "<EOS>":
            break

        parts = token_str.split("_", 1)
        token_type = parts[0]
        token_value_str = parts[1] if len(parts) > 1 else None

        if token_type == "Time" and token_value_str.startswith("Shift"): # Time_Shift_TSX
            bin_idx_str = token_value_str[2:] # Remove "TS"
            try:
                bin_idx = int(bin_idx_str)
                time_shift_b = get_value_from_bin_index(bin_idx, time_shift_bins_beats)
                current_abs_tick += int(time_shift_b * ticks_per_beat)
            except ValueError:
                print(f"Warning: Could not parse time shift bin index from {token_str}")

        elif token_type == "Role":
            active_role = token_value_str
            if active_role not in role_tracks:
                role_tracks[active_role] = mido.MidiTrack()
                mid.tracks.append(role_tracks[active_role])
                # Add track name and program change
                role_tracks[active_role].append(mido.MetaMessage('track_name', name=active_role, time=0))
                program = ROLE_TO_PROGRAM.get(active_role, 0) # Default to Piano
                role_tracks[active_role].append(mido.Message('program_change', program=program, time=0))
                role_last_abs_tick[active_role] = 0 # Initialize last tick for this new track

        elif token_type == "Pitch":
            if active_role:
                try:
                    pending_note_info['pitch'] = int(token_value_str)
                except ValueError:
                    print(f"Warning: Could not parse pitch from {token_str}")
            else: print(f"Warning: Pitch token '{token_str}' encountered without an active role.")


        elif token_type == "Velocity":
            if active_role:
                try:
                    pending_note_info['velocity_bin'] = int(token_value_str)
                except ValueError:
                     print(f"Warning: Could not parse velocity bin from {token_str}")
            else: print(f"Warning: Velocity token '{token_str}' encountered without an active role.")


        elif token_type == "Duration":
            if active_role and 'pitch' in pending_note_info and 'velocity_bin' in pending_note_info:
                try:
                    bin_idx_str = token_value_str[1:] # Remove "D"
                    bin_idx = int(bin_idx_str)
                    duration_b = get_value_from_bin_index(bin_idx, duration_bins_beats)
                    duration_t = int(duration_b * ticks_per_beat)

                    if duration_t <= 0 and duration_b > 0: # If beat duration is positive but ticks are 0 due to rounding
                        duration_t = 1 # Ensure at least 1 tick for very short notes that are not 0 beats

                    if duration_t > 0: # Only add notes with actual duration
                        pitch = pending_note_info['pitch']
                        velocity_bin = pending_note_info['velocity_bin']
                        actual_velocity = map_quantized_velocity_to_midi(velocity_bin, num_velocity_bins)
                        
                        # Calculate delta time for this specific track
                        delta_tick_note_on = current_abs_tick - role_last_abs_tick[active_role]
                        if delta_tick_note_on < 0: delta_tick_note_on = 0 # Should not happen if time always moves forward

                        track = role_tracks[active_role]
                        track.append(mido.Message('note_on', note=pitch, velocity=actual_velocity, time=delta_tick_note_on))
                        track.append(mido.Message('note_off', note=pitch, velocity=0, time=duration_t)) # Duration is the delta for note_off
                        
                        role_last_abs_tick[active_role] = current_abs_tick + duration_t
                    
                    pending_note_info = {} # Clear for next note on this role or other roles
                except ValueError:
                    print(f"Warning: Could not parse duration bin index from {token_str}")
                except KeyError as e:
                    print(f"Warning: Missing pitch or velocity for duration token. {e} Pending: {pending_note_info}")

            # If a duration token appears without full note info, it might be an error in generation
            # or we reset the active_role or pending_note_info
            # elif active_role and ('pitch' not in pending_note_info or 'velocity_bin' not in pending_note_info):
            #    print(f"Warning: Duration token '{token_str}' found but note info incomplete for role {active_role}. Resetting.")
            #    pending_note_info = {}


        elif token_type == "Tempo": # Tempo_TX
            try:
                bin_idx_str = token_value_str[1:] # Remove "T"
                bin_idx = int(bin_idx_str)
                tempo_val_us = int(get_value_from_bin_index(bin_idx, tempo_bins_us))
                
                delta_tick_meta = current_abs_tick - meta_track_last_abs_tick
                if delta_tick_meta < 0: delta_tick_meta = 0
                meta_track.append(mido.MetaMessage('set_tempo', tempo=tempo_val_us, time=delta_tick_meta))
                meta_track_last_abs_tick = current_abs_tick
            except ValueError:
                print(f"Warning: Could not parse tempo bin index from {token_str}")

        elif token_type == "Key": # Key_ActualKeyName
            key_name = token_value_str # e.g., "C", "Am"
            delta_tick_meta = current_abs_tick - meta_track_last_abs_tick
            if delta_tick_meta < 0: delta_tick_meta = 0
            meta_track.append(mido.MetaMessage('key_signature', key=key_name, time=delta_tick_meta))
            meta_track_last_abs_tick = current_abs_tick

        elif token_type == "TimeSig": # TimeSig_N_D
            try:
                num_str, den_str = token_value_str.split('_')
                num = int(num_str)
                den = int(den_str)
                delta_tick_meta = current_abs_tick - meta_track_last_abs_tick
                if delta_tick_meta < 0: delta_tick_meta = 0
                meta_track.append(mido.MetaMessage('time_signature', numerator=num, denominator=den, clocks_per_click=24, notated_32nd_notes_per_beat=8, time=delta_tick_meta))
                meta_track_last_abs_tick = current_abs_tick
            except ValueError:
                print(f"Warning: Could not parse time signature from {token_str}")
    
    # Add end_of_track to all tracks
    # Calculate final delta time for meta track
    final_meta_delta = 0 # if no further events, or some conventional value
    # if mid.length > meta_track_last_abs_tick: # mid.length is tricky, depends on events
    #    final_meta_delta = int(current_abs_tick - meta_track_last_abs_tick) # Approximate
    meta_track.append(mido.MetaMessage('end_of_track', time=max(0, final_meta_delta)))

    for role, track in role_tracks.items():
        final_role_delta = 0
        # if mid.length > role_last_abs_tick[role]:
        #    final_role_delta = int(current_abs_tick - role_last_abs_tick[role]) # Approximate
        track.append(mido.MetaMessage('end_of_track', time=max(0, final_role_delta)))

    try:
        mid.save(output_midi_path)
        print(f"MIDI file saved to {output_midi_path}")
    except Exception as e:
        print(f"Error saving MIDI file: {e}")

    return mid

# %%


# %%

detokenize_to_midi(
            generated_str_sequence,
            ticks_per_beat=TICKS_PER_BEAT_DETOK,
            duration_bins_beats=DURATION_BINS_BEATS,
            time_shift_bins_beats=TIME_SHIFT_BINS_BEATS,
            tempo_bins_us=TEMPO_BINS_US_PER_BEAT,
            num_velocity_bins=NUM_VELOCITY_BINS,
            output_midi_path="output_from_generated_tokens.mid"
        )

# %%


# %%



