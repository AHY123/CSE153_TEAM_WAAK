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
import torch.nn as nn # Will be needed for the model, good to have imports ready
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence # For padding

# --- Assume these are defined from your previous steps ---
# token_to_int = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, ...} # Your full vocabulary
# all_token_sequences_str = [...] # List of lists of string tokens from all your files

# --- 0. Convert String Sequences to Integer Sequences using your new vocab ---
# It's crucial to do this with the vocabulary where <PAD> is 0.
if 'token_to_int' not in globals() or not token_to_int:
    raise ValueError("token_to_int vocabulary is not defined. Please ensure it's loaded or created.")
if '<PAD>' not in token_to_int or token_to_int['<PAD>'] != 0:
    raise ValueError("'<PAD>' token must be in token_to_int and have an ID of 0.")

# Make sure all_token_sequences_str is populated
# For demonstration, let's assume it's populated (replace with your actual data)
# Example: all_token_sequences_str = [
#     ['<SOS>', 'Role_LEAD_PULSE1', 'Pitch_60', 'Velocity_0', 'Duration_D5', '<EOS>'],
#     ['<SOS>', 'Tempo_T0', 'TimeSig_4_4', 'Role_BASS', 'Pitch_40', 'Velocity_0', 'Duration_D10', '<EOS>']
# ]
# if 'all_token_sequences_str' not in globals() or not all_token_sequences_str:
#     print("Warning: all_token_sequences_str is empty. Using dummy data for demonstration.")
#     all_token_sequences_str = [
#          ['<SOS>', 'Role_LEAD_PULSE1', 'Pitch_60', 'Velocity_0', 'Duration_D5', 'Time_Shift_TS0', 'Role_BASS', 'Pitch_40', '<EOS>'],
#          ['<SOS>', 'Tempo_T0', 'Pitch_72', 'Velocity_0', 'Duration_D2', '<EOS>']
#      ]


all_token_sequences_int = []
unknown_token_count = 0
UNK_TOKEN_ID = -1 # Placeholder for unknown, though ideally vocab covers all

if 'all_token_sequences_str' in globals() and all_token_sequences_str: # Check if it exists and is not empty
    for str_seq in all_token_sequences_str:
        int_seq = []
        for token in str_seq:
            idx = token_to_int.get(token)
            if idx is None:
                # This should ideally not happen if vocabulary was built from all sequences
                print(f"Warning: Token '{token}' not in vocabulary! Assigning UNK_TOKEN_ID.")
                int_seq.append(UNK_TOKEN_ID) # Or handle error, or add <UNK> to vocab
                unknown_token_count += 1
            else:
                int_seq.append(idx)
        all_token_sequences_int.append(int_seq)
    if unknown_token_count > 0:
        print(f"Total unknown tokens encountered: {unknown_token_count}")
else:
    print("Error: `all_token_sequences_str` is not defined or empty. Cannot proceed to create integer sequences.")
    # You might want to stop execution here or load your data


# --- 1. Configuration for Data Preparation ---
pad_idx = token_to_int["<PAD>"] # Should be 0
sos_idx = token_to_int["<SOS>"] # Should be 1
eos_idx = token_to_int["<EOS>"] # Should be 2

# IMPORTANT: Choose MAX_SEQUENCE_LENGTH based on your dataset analysis
# (e.g., 95th percentile of sequence lengths after tokenization, including SOS/EOS)
# Sequences longer than this will be truncated for input to the model.
# The model will learn to predict up to this length.
MAX_SEQUENCE_LENGTH = 256 # Example: For model input x (target y will also be this length)

BATCH_SIZE = 32         # Adjust based on your GPU memory and dataset size


# --- 2. Define PyTorch Dataset ---
class MusicDataset(Dataset):
    def __init__(self, integer_sequences, max_model_input_len):
        self.sequences = []
        # max_model_input_len is the length for x and y (i.e., MAX_SEQUENCE_LENGTH above)
        # The original sequence can be up to max_model_input_len + 1 to derive x and y.

        for seq_tensor in integer_sequences:
            if len(seq_tensor) < 2: # Needs at least SOS and one data point (to make x and y) or SOS+EOS
                continue
            # Truncate original sequence if it's too long to form x and y of max_model_input_len
            self.sequences.append(seq_tensor[:max_model_input_len + 1])

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        # Input is sequence[:-1], Target is sequence[1:]
        # Both x and y will have length up to MAX_SEQUENCE_LENGTH
        x = seq[:-1]
        y = seq[1:]
        return x, y

# --- 3. Define Collate Function for Padding ---
def collate_fn(batch, pad_idx, max_len):
    inputs, targets = zip(*batch)

    # Pad inputs (x) to max_len
    inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=pad_idx)
    # Ensure consistent length due to potential variations before padding if some original sequences were shorter than max_len+1
    if inputs_padded.size(1) < max_len:
        padding = torch.full((inputs_padded.size(0), max_len - inputs_padded.size(1)), 
                             pad_idx, dtype=torch.long, device=inputs_padded.device)
        inputs_padded = torch.cat([inputs_padded, padding], dim=1)
    elif inputs_padded.size(1) > max_len:
        inputs_padded = inputs_padded[:, :max_len]


    # Pad targets (y) to max_len
    targets_padded = pad_sequence(targets, batch_first=True, padding_value=pad_idx)
    if targets_padded.size(1) < max_len:
        padding = torch.full((targets_padded.size(0), max_len - targets_padded.size(1)), 
                             pad_idx, dtype=torch.long, device=targets_padded.device)
        targets_padded = torch.cat([targets_padded, padding], dim=1)
    elif targets_padded.size(1) > max_len:
        targets_padded = targets_padded[:, :max_len]
        
    return inputs_padded, targets_padded

# --- 4. Create Dataset and DataLoaders ---
if all_token_sequences_int: # Proceed only if we have integer sequences
    # Convert sequences to Tensors first for MusicDataset
    tensor_sequences = [torch.tensor(seq, dtype=torch.long) for seq in all_token_sequences_int]

    full_dataset = MusicDataset(tensor_sequences, MAX_SEQUENCE_LENGTH)
    print(f"\nTotal sequences in full_dataset: {len(full_dataset)}")

    if len(full_dataset) > 0:
        # Split into training and validation
        # Adjust split ratio as needed, e.g., 90% train, 10% val
        train_size = int(0.9 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        
        # Handle cases where dataset is too small for split
        if val_size == 0 and train_size > 0: # If only enough for train
            train_dataset = full_dataset
            val_dataset = None # Or copy a small part of train_dataset for validation
            print("Warning: Dataset too small for a validation split. Using all data for training.")
        elif train_size == 0:
             raise ValueError("Dataset is too small to create even a training set.")
        else:
            train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

        print(f"Train dataset size: {len(train_dataset)}")
        if val_dataset:
            print(f"Validation dataset size: {len(val_dataset)}")

        # Create DataLoaders
        # The collate_fn needs pad_idx and MAX_SEQUENCE_LENGTH
        custom_collate_fn = lambda b: collate_fn(b, pad_idx, MAX_SEQUENCE_LENGTH)

        train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, 
                                      shuffle=True, collate_fn=custom_collate_fn)
        if val_dataset:
            val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, 
                                        shuffle=False, collate_fn=custom_collate_fn)
        else:
            val_dataloader = None

        print(f"\nCreated DataLoaders. Batch size: {BATCH_SIZE}")

        # --- 5. Test one batch (Optional) ---
        if train_dataloader:
            print("\n--- Sample Batch from Training DataLoader ---")
            try:
                x_sample_batch, y_sample_batch = next(iter(train_dataloader))
                print(f"x_sample_batch shape: {x_sample_batch.shape}") # Should be (BATCH_SIZE, MAX_SEQUENCE_LENGTH)
                print(f"y_sample_batch shape: {y_sample_batch.shape}") # Should be (BATCH_SIZE, MAX_SEQUENCE_LENGTH)
                print(f"Example x sequence (first in batch, first 10 tokens): {x_sample_batch[0, :10]}")
                print(f"Example y sequence (first in batch, first 10 tokens): {y_sample_batch[0, :10]}")
            except StopIteration:
                print("Training Dataloader is empty. This might happen if train_size was 0.")
            except Exception as e:
                print(f"Error getting sample batch: {e}")
    else:
        print("MusicDataset is empty after processing. Cannot create DataLoaders.")
        print("This might be due to all sequences being too short (<2 tokens).")

else:
    print("No integer sequences available (`all_token_sequences_int` is empty). Cannot create Dataset and DataLoaders.")
    print("Please ensure your tokenization step produced string sequences and they were converted to integers.")

# %%
all_token_sequences_int

# %%
import torch
import torch.nn as nn
import math

# --- Assume these are defined from your previous steps ---
# vocab_size = 126 # From your vocabulary
# MAX_SEQUENCE_LENGTH = 256 # From your data preparation

# --- 1. Positional Encoding ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe) # So it moves to device with the model, but not a parameter

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

# --- 2. Transformer Model (Decoder-Only) ---
class MusicTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_decoder_layers, dim_feedforward, dropout=0.1, max_seq_len=512): # max_seq_len for PositionalEncoding
        super(MusicTransformer, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=max_seq_len)
        
        # Using nn.TransformerDecoderLayer and nn.TransformerDecoder for a decoder-only setup
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout, 
            batch_first=True, # Important: expects (batch, seq, feature)
            activation='gelu' # GELU is often used in modern Transformers
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, 
            num_layers=num_decoder_layers
        )
        
        self.fc_out = nn.Linear(d_model, vocab_size)

    def _generate_square_subsequent_mask(self, sz, device):
        """Generates a square causal mask for the sequence. (sz x sz)"""
        mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, src_padding_mask=None):
        """
        Args:
            src: Tensor, shape [batch_size, seq_len] (input integer token sequence)
            src_padding_mask: Tensor, shape [batch_size, seq_len] (True where src is padded)
        """
        seq_len = src.size(1)
        
        # Create causal mask (aka look-ahead mask or subsequent mask)
        # This prevents positions from attending to subsequent positions.
        # Shape: (seq_len, seq_len)
        tgt_mask = self._generate_square_subsequent_mask(seq_len, src.device)

        # Embedding and Positional Encoding
        embedded_src = self.embedding(src) * math.sqrt(self.d_model) # Scale embedding
        pos_encoded_src = self.pos_encoder(embedded_src) # Shape: [batch_size, seq_len, d_model]
        
        # For a decoder-only Transformer, the 'target' input to nn.TransformerDecoder
        # is the sequence itself (pos_encoded_src).
        # The 'memory' input is also the sequence itself.
        # tgt_mask is the causal mask.
        # memory_mask can be the same as tgt_mask or None if memory is same as tgt in self-attention.
        # tgt_key_padding_mask and memory_key_padding_mask are derived from src_padding_mask.
        
        output = self.transformer_decoder(
            tgt=pos_encoded_src,             # Target sequence (what the decoder processes)
            memory=pos_encoded_src,          # Memory sequence (what the decoder attends to from encoder - here it's self-attention)
            tgt_mask=tgt_mask,               # Causal mask for target self-attention
            memory_mask=None,                # No separate memory mask needed if memory is tgt & masked by tgt_mask for self-attention
                                             # OR pass tgt_mask here too if required by implementation details for self-reference
            tgt_key_padding_mask=src_padding_mask, # Padding mask for target sequence
            memory_key_padding_mask=src_padding_mask # Padding mask for memory sequence
        )
        # Output shape: [batch_size, seq_len, d_model]
        
        logits = self.fc_out(output) # Project to vocabulary size
        # Logits shape: [batch_size, seq_len, vocab_size]
        return logits

# --- 3. Define Hyperparameters and Instantiate the Model ---

# You should have these from your data preparation:
# vocab_size = 126  # Replace with your actual vocab_size
# MAX_SEQUENCE_LENGTH = 256 # Replace with your actual MAX_SEQUENCE_LENGTH

# Model Hyperparameters (These are starting points, you'll likely need to tune them)
D_MODEL = 256           # Embedding dimension and model dimension.
                        # For vocab 126, 256 is a reasonable start. Could try 128 or 512.
NHEAD = 4               # Number of attention heads. Must be a divisor of D_MODEL (256 % 4 == 0).
NUM_DECODER_LAYERS = 4  # Number of Transformer decoder blocks (e.g., 3-6 is common).
DIM_FEEDFORWARD = 1024  # Dimension of the feed-forward network inside Transformer blocks.
                        # Often 2*D_MODEL or 4*D_MODEL (1024 = 4 * 256).
DROPOUT = 0.1           # Dropout rate.

# Determine device
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA (NVIDIA GPU)")
# Check for MPS (Apple Silicon GPU on macOS)
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS (Apple Silicon GPU)")
else:
    device = torch.device("cpu")
    print("Using CPU")
print(f"Using device: {device}")

# # --- Instantiate the model (Uncomment when you have vocab_size and MAX_SEQUENCE_LENGTH defined) ---
# # Ensure vocab_size and MAX_SEQUENCE_LENGTH are correctly defined from your previous steps!
# if 'vocab_size' not in globals() or 'MAX_SEQUENCE_LENGTH' not in globals():
#     print("Please define 'vocab_size' and 'MAX_SEQUENCE_LENGTH' before instantiating the model.")
#     print("Example: vocab_size = 126; MAX_SEQUENCE_LENGTH = 256")
# else:
#     model = MusicTransformer(
#         vocab_size=vocab_size,
#         d_model=D_MODEL,
#         nhead=NHEAD,
#         num_decoder_layers=NUM_DECODER_LAYERS,
#         dim_feedforward=DIM_FEEDFORWARD,
#         dropout=DROPOUT,
#         max_seq_len=MAX_SEQUENCE_LENGTH # For PositionalEncoding buffer size
#     ).to(device)

#     print("\n--- Model Architecture ---")
#     print(model)
#     total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     print(f"\nTotal Trainable Parameters: {total_params:,}")

#     # --- Test with a dummy batch (Optional, requires DataLoaders from Step 1) ---
#     # This test assumes you have train_dataloader defined and populated
#     # if 'train_dataloader' in globals() and train_dataloader:
#     #     try:
#     #         print("\n--- Testing model with a dummy batch ---")
#     #         dummy_x_batch, _ = next(iter(train_dataloader))
#     #         dummy_x_batch = dummy_x_batch.to(device)
#     #         dummy_padding_mask = (dummy_x_batch == pad_idx).to(device) # Assuming pad_idx is defined
            
#     #         with torch.no_grad():
#     #             logits = model(dummy_x_batch, src_padding_mask=dummy_padding_mask)
#     #         print(f"Output logits shape: {logits.shape}") # Expected: (BATCH_SIZE, MAX_SEQUENCE_LENGTH, VOCAB_SIZE)
#     #     except NameError as e:
#     #         print(f"Could not run dummy batch test: {e}. Ensure train_dataloader and pad_idx are defined.")
#     #     except Exception as e:
#     #         print(f"Error during dummy batch test: {e}")
#     # else:
#     #     print("\nSkipping dummy batch test as train_dataloader is not available.")

# %%



