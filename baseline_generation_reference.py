# %% [markdown]
# ## Homework 3: Symbolic Music Generation Using Markov Chains

# %% [markdown]
# **Before starting the homework:**
# 
# Please run `pip install miditok` to install the [MiDiTok](https://github.com/Natooz/MidiTok) package, which simplifies MIDI file processing by making note and beat extraction more straightforward.
# 
# You’re also welcome to experiment with other MIDI processing libraries such as [mido](https://github.com/mido/mido), [pretty_midi](https://github.com/craffel/pretty-midi) and [miditoolkit](https://github.com/YatingMusic/miditoolkit). However, with these libraries, you’ll need to handle MIDI quantization yourself, for example, converting note-on/note-off events into beat positions and durations.

# %%
# install some packages
# !pip install miditok
# !pip install symusic

# %%
# import required packages
import random
# random.seed(42)
from glob import glob
from collections import defaultdict

import numpy as np
from numpy.random import choice

from symusic import Score
from miditok import REMI, TokenizerConfig
from midiutil import MIDIFile
import pandas as pd

# %% [markdown]
# ### Load music dataset
# We use a subset of [PDMX dataset](https://zenodo.org/records/14984509) for this homework. 
# 
# Please download the data through XXXXX and unzip.
# 
# All pieces are monophonic music (i.e. one melody line) in time signature 4/4.

# %%
midi_files = glob('PDMX_subset/*.mid')
len(midi_files)

# %% [markdown]
# ### Train a tokenizer with the REMI method in MidiTok

# %%
config = TokenizerConfig(num_velocities=1, use_chords=False, use_programs=False)
tokenizer = REMI(config)
tokenizer.train(vocab_size=1000, files_paths=midi_files)

# %% [markdown]
# ### Use the trained tokenizer to get tokens for each midi file
# In REMI representation, each note will be represented with four tokens: `Position, Pitch, Velocity, Duration`, e.g. `('Position_28', 'Pitch_74', 'Velocity_127', 'Duration_0.4.8')`, and `Bar_None` token indicates the beginning of a new bar.

# %%
# midi = Score(midi_files[0])
# tokens = tokenizer(midi)[0].tokens
# tokens[:10]
df = pd.read_csv('data/all_more_than_20.csv')
filenames = df[["file_name"]].assign(p='data/nesmdb_midi/all/' + df['file_name'])['p'].to_list()
config = TokenizerConfig(num_velocities=1, use_chords=False, use_programs=False)
tokenizer = REMI(config)
tokenizer.train(vocab_size=1000, files_paths=filenames)
midi = Score(filenames[0])
tokens = tokenizer(midi)[0].tokens
tokens[:10]

# %% [markdown]
# 1. Write a function to extract note pitch events from a midi file; extract all note pitch events from the dataset and output a dictionary that maps note pitch events to the number of times they occur in the files. (e.g. {60: 120, 61: 58, …}).
# 
# `note_extraction()`
# - **Input**: a midi file
# 
# - **Output**: a list of note pitch events
# 
# `note_frequency()`
# - **Input**: all midi files `midi_files`
# 
# - **Output**: a dictionary that maps note pitch events to the number of times they occur, e.g {60: 120, 61: 58, …}

# %%
def note_extraction(midi_file):
    # Q1a: Your code goes here
    note_events = []
    midi = Score(midi_file)
    tokens = tokenizer(midi)[0].tokens
    for token in tokens:
        if 'Pitch' in token:
            note = int(token.split('_')[1])
            note_events.append(note)
    return note_events

# %%
def note_frequency(midi_files):
    # Q1b: Your code goes here
    note_counts = defaultdict(int)
    for midi_file in midi_files:
        note_events = note_extraction(midi_file)
        for note in note_events:
            note_counts[note] += 1
    return note_counts

# %% [markdown]
# 2. Write a function to normalize the above dictionary to produce probability scores. (e.g. {60: 0.13, 61: 0.065, …})
# 
# `note_unigram_probability()`
# - **Input**: all midi files `midi_files`
# 
# - **Output**: a dictionary that maps note pitch events to probabilities they occur in the dataset, e.g. {60: 0.13, 61: 0.06, …}

# %%
def note_unigram_probability(midi_files):
    note_counts = note_frequency(midi_files)
    
    # Q2: Your code goes here
    unigramProbabilities = {}
    counts = sum(list(note_counts.values()))
    for n in note_counts:
        unigramProbabilities[n] = note_counts[n] / counts
    return unigramProbabilities

# %% [markdown]
# 3. Generate a table of pairwise probabilities containing p(next_note | previous_note) for the dataset; write a function that randomly generates the next note based on the previous note based on this distribution.
# 
# `note_bigram_probability()`
# - **Input**: all midi files `midi_files`
# 
# - **Output**: two dictionaries:
# 
#   - `bigramTransitions`: key - previous_note, value - a list of next_note, e.g. {60:[62, 64, ..], 62:[60, 64, ..], ...}
# 
#   - `bigramTransitionProbabilities`: key - previous_note, value - a list of probabilities for next_note in the same order of `bigramTransitions`, e.g. {60:[0.3, 0.4, ..], 62:[0.2, 0.1, ..], ...}
# 
# `sample_next_note()`
# - **Input**: a note
# 
# - **Output**: next note sampled from pairwise probabilities

# %%
def note_bigram_probability(midi_files):
    # Q3a: Your code goes here
    bigrams = defaultdict(int)
    
    for file in midi_files:
        note_events = note_extraction(file)
        for (note1, note2) in zip(note_events[:-1], note_events[1:]):
            bigrams[(note1, note2)] += 1
            
    bigramTransitions = defaultdict(list)
    bigramTransitionProbabilities = defaultdict(list)

    for b1,b2 in bigrams:
        bigramTransitions[b1].append(b2)
        bigramTransitionProbabilities[b1].append(bigrams[(b1,b2)])
        
    for k in bigramTransitionProbabilities:
        Z = sum(bigramTransitionProbabilities[k])
        bigramTransitionProbabilities[k] = [x / Z for x in bigramTransitionProbabilities[k]]
        
    return bigramTransitions, bigramTransitionProbabilities

# %%
def sample_next_note(note):
    # Q3b: Your code goes here
    bigramTransitions, bigramTransitionProbabilities = note_bigram_probability(midi_files)
    next_note = choice(bigramTransitions[note], 1, p=bigramTransitionProbabilities[note])[0]
    return next_note

# %% [markdown]
# 4. Write a function to calculate the perplexity of your model on a midi file.
# 
#     The perplexity of a model is defined as 
# 
#     $\quad \text{exp}(-\frac{1}{N} \sum_{i=1}^N \text{log}(p(w_i|w_{i-1})))$
# 
#     where $p(w_1|w_0) = p(w_1)$, $p(w_i|w_{i-1}) (i>1)$ refers to the pairwise probability p(next_note | previous_note).
# 
# `note_bigram_perplexity()`
# - **Input**: a midi file
# 
# - **Output**: perplexity value

# %%
def note_bigram_perplexity(midi_file):
    unigramProbabilities = note_unigram_probability(midi_files)
    bigramTransitions, bigramTransitionProbabilities = note_bigram_probability(midi_files)
    
    # Q4: Your code goes here
    note_events = note_extraction(midi_file)
    perplexities = [unigramProbabilities[note_events[0]]]
    for (note1, note2) in zip(note_events[:-1], note_events[1:]):
        index = bigramTransitions[note1].index(note2)
        prob = bigramTransitionProbabilities[note1][index]
        perplexities.append(prob)

    assert len(perplexities) == len(note_events)
    perplexity = np.exp(-np.sum(np.log(perplexities)) / len(note_events))
    return perplexity

# %% [markdown]
# 5. Implement a second-order Markov chain, i.e., one which estimates p(next_note | next_previous_note, previous_note); write a function to compute the perplexity of this new model on a midi file. 
# 
#     The perplexity of this model is defined as 
# 
#     $\quad \text{exp}(-\frac{1}{N} \sum_{i=1}^N \text{log}(p(w_i|w_{i-2}, w_{i-1})))$
# 
#     where $p(w_1|w_{-1}, w_0) = p(w_1)$, $p(w_2|w_0, w_1) = p(w_2|w_1)$, $p(w_i|w_{i-2}, w_{i-1}) (i>2)$ refers to the probability p(next_note | next_previous_note, previous_note).
# 
# 
# `note_trigram_probability()`
# - **Input**: all midi files `midi_files`
# 
# - **Output**: two dictionaries:
# 
#   - `trigramTransitions`: key - (next_previous_note, previous_note), value - a list of next_note, e.g. {(60, 62):[64, 66, ..], (60, 64):[60, 64, ..], ...}
# 
#   - `trigramTransitionProbabilities`: key - (next_previous_note, previous_note), value - a list of probabilities for next_note in the same order of `trigramTransitions`, e.g. {(60, 62):[0.2, 0.2, ..], (60, 64):[0.4, 0.1, ..], ...}
# 
# `note_trigram_perplexity()`
# - **Input**: a midi file
# 
# - **Output**: perplexity value

# %%
def note_trigram_probability(midi_files):
    # Q5a: Your code goes here
    trigrams = defaultdict(int)
    for file in midi_files:
        note_events = note_extraction(file)
        for (note1, note2, note3) in zip(note_events[:-2], note_events[1:-1], note_events[2:]):
            trigrams[(note1, note2, note3)] += 1
            
    trigramTransitions = defaultdict(list)
    trigramTransitionProbabilities = defaultdict(list)

    for t1,t2,t3 in trigrams:
        trigramTransitions[(t1,t2)].append(t3)
        trigramTransitionProbabilities[(t1,t2)].append(trigrams[(t1,t2,t3)])
        
    for k in trigramTransitionProbabilities:
        Z = sum(trigramTransitionProbabilities[k])
        trigramTransitionProbabilities[k] = [x / Z for x in trigramTransitionProbabilities[k]]
        
    return trigramTransitions, trigramTransitionProbabilities

# %%
def note_trigram_perplexity(midi_file):
    unigramProbabilities = note_unigram_probability(midi_files)
    bigramTransitions, bigramTransitionProbabilities = note_bigram_probability(midi_files)
    trigramTransitions, trigramTransitionProbabilities = note_trigram_probability(midi_files)
    
    # Q5b: Your code goes here
    note_events = note_extraction(midi_file)
    perplexities = [unigramProbabilities[note_events[0]]]
    index = bigramTransitions[note_events[0]].index(note_events[1])
    prob = bigramTransitionProbabilities[note_events[0]][index]
    perplexities.append(prob)
    
    for (note1, note2, note3) in zip(note_events[:-2], note_events[1:-1], note_events[2:]):
        index = trigramTransitions[(note1, note2)].index(note3)
        prob = trigramTransitionProbabilities[(note1, note2)][index]
        perplexities.append(prob)

    assert len(perplexities) == len(note_events)
    perplexity = np.exp(-np.sum(np.log(perplexities)) / len(note_events))
    return perplexity

# %% [markdown]
# 6. Our model currently doesn’t have any knowledge of beats. Write a function that extracts beat lengths and outputs a list of [(beat position; beat length)] values.
# 
#     Recall that each note will be encoded as `Position, Pitch, Velocity, Duration` using REMI. Please keep the `Position` value for beat position, and convert `Duration` to beat length using provided lookup table `duration2length` (see below).
# 
#     For example, for a note represented by four tokens `('Position_24', 'Pitch_72', 'Velocity_127', 'Duration_0.4.8')`, the extracted (beat position; beat length) value is `(24, 4)`.
# 
#     As a result, we will obtain a list like [(0,8),(8,16),(24,4),(28,4),(0,4)...], where the next beat position is the previous beat position + the beat length. As we divide each bar into 32 positions by default, when reaching the end of a bar (i.e. 28 + 4 = 32 in the case of (28, 4)), the beat position reset to 0.

# %%
duration2length = {
    '0.1.8': 1,
    '0.2.8': 2,
    '0.3.8': 3,
    '0.4.8': 4,
    '0.5.8': 5,
    '0.6.8': 6,
    '0.7.8': 7,
    '1.0.8': 8,
    '1.1.8': 9,
    '1.2.8': 10,
    '1.3.8': 11,
    '1.4.8': 12,
    '1.5.8': 13,
    '1.6.8': 14,
    '1.7.8': 15,
    '2.0.8': 16,
    '2.1.8': 17,
    '2.2.8': 18,
    '2.3.8': 19,
    '2.4.8': 20,
    '2.5.8': 21,
    '2.6.8': 22,
    '2.7.8': 23,
    '3.0.8': 24,
    '3.1.8': 25,
    '3.2.8': 26,
    '3.3.8': 27,
    '3.4.8': 28,
    '3.5.8': 29,
    '3.6.8': 30,
    '3.7.8': 31,
    '4.0.4': 32,
    '4.1.4': 34,
    '4.2.4': 36,
    '4.3.4': 38,
    '5.0.4': 40,
    '5.1.4': 42,
    '5.2.4': 44,
    '5.3.4': 46,
    '6.0.4': 48,
    '6.1.4': 50,
    '6.2.4': 52,
    '6.3.4': 54,
    '7.0.4': 56,
    '7.1.4': 58,
    '7.2.4': 60,
    '7.3.4': 62,
    '8.0.4': 64,
    '8.1.4': 66,
    '8.2.4': 68,
    '8.3.4': 70,
    '9.0.4': 72,
    '9.1.4': 74,
    '9.2.4': 76,
    '9.3.4': 78,
    '10.0.4': 80,
    '10.1.4': 82,
    '10.2.4': 84,
    '10.3.4': 86,
    '11.0.4': 88,
    '11.1.4': 90,
    '11.2.4': 92,
    '11.3.4': 94,
    '12.0.4': 96
}

# %% [markdown]
# `beat_extraction()`
# - **Input**: a midi file
# 
# - **Output**: a list of (beat position; beat length) values

# %%
def beat_extraction(midi_file):
    # Q6: Your code goes here
    midi = Score(midi_file)
    tokens = tokenizer(midi)[0].tokens
    beats = []
    # unique = set()
    
    for i in range(len(tokens)):
        if 'Position' in tokens[i] and 'Duration' in tokens[i+3]:
            position = int(tokens[i].split('_')[1])
            # unique.add(tokens[i+3].split('_')[1])
            length = duration2length[tokens[i+3].split('_')[1]]
            beats.append((position, length))
    return beats
    # return unique

# %% [markdown]
# 7. Implement a Markov chain that computes p(beat_length | previous_beat_length) based on the above function.
# 
# `beat_bigram_probability()`
# - **Input**: all midi files `midi_files`
# 
# - **Output**: two dictionaries:
# 
#   - `bigramBeatTransitions`: key - previous_beat_length, value - a list of beat_length, e.g. {4:[8, 2, ..], 8:[8, 4, ..], ...}
# 
#   - `bigramBeatTransitionProbabilities`: key - previous_beat_length, value - a list of probabilities for beat_length in the same order of `bigramBeatTransitions`, e.g. {4:[0.3, 0.2, ..], 8:[0.4, 0.4, ..], ...}

# %%
def beat_bigram_probability(midi_files):
    # Q7: Your code goes here
    bigramBeat = defaultdict(int)
    for file in midi_files:
        beats = beat_extraction(file)
        for (beat1, beat2) in zip(beats[:-1], beats[1:]):
            bigramBeat[(beat1[1], beat2[1])] += 1
            
    bigramBeatTransitions = defaultdict(list)
    bigramBeatTransitionProbabilities = defaultdict(list)

    for b1,b2 in bigramBeat:
        bigramBeatTransitions[b1].append(b2)
        bigramBeatTransitionProbabilities[b1].append(bigramBeat[(b1,b2)])
        
    for k in bigramBeatTransitionProbabilities:
        Z = sum(bigramBeatTransitionProbabilities[k])
        bigramBeatTransitionProbabilities[k] = [x / Z for x in bigramBeatTransitionProbabilities[k]]
        
    return bigramBeatTransitions, bigramBeatTransitionProbabilities

# %% [markdown]
# 8. Implement a function to compute p(beat length | beat position), and compute the perplexity of your models from Q7 and Q8. For both models, we only consider the probabilities of predicting the sequence of **beat length**.
# 
# `beat_pos_bigram_probability()`
# - **Input**: all midi files `midi_files`
# 
# - **Output**: two dictionaries:
# 
#   - `bigramBeatPosTransitions`: key - beat_position, value - a list of beat_length
# 
#   - `bigramBeatPosTransitionProbabilities`: key - beat_position, value - a list of probabilities for beat_length in the same order of `bigramBeatPosTransitions`
# 
# `beat_bigram_perplexity()`
# - **Input**: a midi file
# 
# - **Output**: two perplexity values correspond to the models in Q7 and Q8, respectively

# %%
def beat_pos_bigram_probability(midi_files):
    # Q8a: Your code goes here
    bigramBeatPos = defaultdict(int)
    for file in midi_files:
        beats = beat_extraction(file)
        for beat in beats:
            bigramBeatPos[(beat[0], beat[1])] += 1
            
    bigramBeatPosTransitions = defaultdict(list)
    bigramBeatPosTransitionProbabilities = defaultdict(list)

    for b1,b2 in bigramBeatPos:
        bigramBeatPosTransitions[b1].append(b2)
        bigramBeatPosTransitionProbabilities[b1].append(bigramBeatPos[(b1,b2)])
        
    for k in bigramBeatPosTransitionProbabilities:
        Z = sum(bigramBeatPosTransitionProbabilities[k])
        bigramBeatPosTransitionProbabilities[k] = [x / Z for x in bigramBeatPosTransitionProbabilities[k]]
        
    return bigramBeatPosTransitions, bigramBeatPosTransitionProbabilities

# %%
def beat_bigram_perplexity(midi_file):
    bigramBeatTransitions, bigramBeatTransitionProbabilities = beat_bigram_probability(midi_files)
    bigramBeatPosTransitions, bigramBeatPosTransitionProbabilities = beat_pos_bigram_probability(midi_files)
    # Q8b: Your code goes here
    # Hint: one more probability function needs to be computed
    unigramBeat = defaultdict(int)
    for file in midi_files:
        beats = beat_extraction(file)
        for beat in beats:
            unigramBeat[beat[1]] += 1
    unigramBeatProbabilities = {}
    counts = sum(list(unigramBeat.values()))
    for n in unigramBeat:
        unigramBeatProbabilities[n] = unigramBeat[n] / counts
        
    beat_events = beat_extraction(midi_file)
    beats = [b[1] for b in beat_events]

    # perplexity for Q7
    perplexities = [unigramBeatProbabilities[beats[0]]]
    for (beat1, beat2) in zip(beats[:-1], beats[1:]):
        index = bigramBeatTransitions[beat1].index(beat2)
        prob = bigramBeatTransitionProbabilities[beat1][index]
        perplexities.append(prob)
    assert len(perplexities) == len(beats)
    perplexity_Q7 = np.exp(-np.sum(np.log(perplexities)) / len(beats))
    
    # perplexity for Q8
    perplexities = []
    for (beat_position, beat_length) in beat_events:
        index = bigramBeatPosTransitions[beat_position].index(beat_length)
        prob = bigramBeatPosTransitionProbabilities[beat_position][index]
        perplexities.append(prob)
    assert len(perplexities) == len(beat_events)
    perplexity_Q8 = np.exp(-np.sum(np.log(perplexities)) / len(beats))
    
    return perplexity_Q7, perplexity_Q8

# %% [markdown]
# 9. Implement a Markov chain that computes p(beat_length | previous_beat_length, beat_position), and report its perplexity. 
# 
# `beat_trigram_probability()`
# - **Input**: all midi files `midi_files`
# 
# - **Output**: two dictionaries:
# 
#   - `trigramBeatTransitions`: key - (previous_beat_length, beat_position), value - a list of beat_length
# 
#   - `trigramBeatTransitionProbabilities`: key: (previous_beat_length, beat_position), value: a list of probabilities for beat_length in the same order of `trigramsBeatTransition`
# 
# `beat_trigram_perplexity()`
# - **Input**: a midi file
# 
# - **Output**: perplexity value

# %%
def beat_trigram_probability(midi_files):
    # Q9a: Your code goes here
    trigramBeat = defaultdict(int)
    for file in midi_files:
        beats = beat_extraction(file)
        for (beat1, beat2) in zip(beats[:-1], beats[1:]):
            trigramBeat[(beat1[1], beat2[0], beat2[1])] += 1
            
    trigramBeatTransitions = defaultdict(list)
    trigramBeatTransitionProbabilities = defaultdict(list)

    for t1,t2,t3 in trigramBeat:
        trigramBeatTransitions[(t1,t2)].append(t3)
        trigramBeatTransitionProbabilities[(t1,t2)].append(trigramBeat[(t1,t2,t3)])
        
    for k in trigramBeatTransitionProbabilities:
        Z = sum(trigramBeatTransitionProbabilities[k])
        trigramBeatTransitionProbabilities[k] = [x / Z for x in trigramBeatTransitionProbabilities[k]]
        
    return trigramBeatTransitions, trigramBeatTransitionProbabilities

# %%
def beat_trigram_perplexity(midi_file):
    bigramBeatPosTransitions, bigramBeatPosTransitionProbabilities = beat_pos_bigram_probability(midi_files)
    trigramBeatTransitions, trigramBeatTransitionProbabilities = beat_trigram_probability(midi_files)
    # Q9b: Your code goes here
    beats = beat_extraction(midi_file)

    perplexities = []
    index = bigramBeatPosTransitions[beats[0][0]].index(beats[0][1])
    prob = bigramBeatPosTransitionProbabilities[beats[0][0]][index]
    perplexities.append(prob)

    for (beat1, beat2) in zip(beats[:-1], beats[1:]):
        index = trigramBeatTransitions[(beat1[1], beat2[0])].index(beat2[1])
        prob = trigramBeatTransitionProbabilities[(beat1[1], beat2[0])][index]
        perplexities.append(prob)

    assert len(perplexities) == len(beats)
    perplexity = np.exp(-np.sum(np.log(perplexities)) / len(beats))
    return perplexity

# %% [markdown]
# 10. Use the model from Q5 to generate 500 notes, and the model from Q8 to generate beat lengths for each note. Save the generated music as a midi file (see code from workbook1) as q10.mid. Remember to reset the beat position to 0 when reaching the end of a bar.
# 
# `music_generate`
# - **Input**: target length, e.g. 500
# 
# - **Output**: a midi file q10.mid
# 
# Note: the duration of one beat in MIDIUtil is 1, while in MidiTok is 8. Divide beat length by 8 if you use methods in MIDIUtil to save midi files.

# %%
def music_generate(length, midi_files, uni=None, bi=None, tri=None):
    # sample notes
    if uni:
        unigramProbabilities = uni
    else:
        unigramProbabilities = note_unigram_probability(midi_files)
    print('unigram done')

    if bi:
        bigramTransitions, bigramTransitionProbabilities = bi
    else:
        bigramTransitions, bigramTransitionProbabilities = note_bigram_probability(midi_files)
    print('bigram done')

    if tri:
        trigramTransitions, trigramTransitionProbabilities = tri
    else:
        trigramTransitions, trigramTransitionProbabilities = note_trigram_probability(midi_files)
    print('trigram done')
    
    # Your code goes here ...
    first_note = choice(list(unigramProbabilities.keys()), 1, p=list(unigramProbabilities.values())).item()
    second_note = choice(bigramTransitions[first_note], 1, p=bigramTransitionProbabilities[first_note]).item()
    sampled_notes = [first_note, second_note]
    while len(sampled_notes) < length:
        next_note = choice(trigramTransitions[(sampled_notes[-2], sampled_notes[-1])], 1, 
                            p=trigramTransitionProbabilities[(sampled_notes[-2], sampled_notes[-1])])
        sampled_notes.append(next_note.item())
    
    # sample beats
    bigramBeatPosTransitions, bigramBeatPosTransitionProbabilities = beat_pos_bigram_probability(midi_files)
    first_beat = choice(bigramBeatPosTransitions[0], 1, p=bigramBeatPosTransitionProbabilities[0]).item()
    sampled_beats = [(0, first_beat)]
    while len(sampled_beats) < length:
        beat_position = sum(sampled_beats[-1]) % 32
        beat_length = choice(bigramBeatPosTransitions[beat_position], 1, 
                        p=bigramBeatPosTransitionProbabilities[beat_position]).item()
        sampled_beats.append((beat_position, beat_length))
    sampled_beats = [beat[1] / 8 for beat in sampled_beats]
    
    # save the generated music as a midi file
    midi = MIDIFile(1)
    track = 0
    time = 0
    tempo = 120
    midi.addTempo(track, time, tempo)
    
    current_time = 0
    for pitch, duration in zip(sampled_notes, sampled_beats):
        midi.addNote(track, 0, pitch, current_time, duration, 100)
        current_time += duration
    with open("q10.mid", "wb") as f:
        midi.writeFile(f)

# %%



