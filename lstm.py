import numpy as np
import os
import pickle
import random
import time
import torch

from music21 import converter, chord, duration, instrument, midi, note, stream
from tqdm import tqdm

# Constants
data_dir = "data/RNN/" # Where is your data located
save_dir = "data/RNN/saved_data" # Where should your outputs and model weights be saved

batch_size = 64 # Size of batches
bidirectional = True # If the LSTM is bidirectional or not
dropout = .3 # Dropout value to use for LSTM
embedding_size = 100 # Embedding size
epochs = 50 # Epochs of training
hidden_size = 256 # Size of the hidden state
lr = .001 # The learning rate
num_layers = 2 # The number of LSTM layers
seq_length = 32 # Length of inputs to the RNN

class SongData(torch.utils.data.Dataset):
    """
    A dataset class for song pitches and note lengths.
    """

    def __init__(self, indices, x_pitches, x_note_lengths, y_pitches, y_pitch_lengths):
        """
        Takes in the data and labels for both pitches and note lengths as well as a list of indices. Indices correspond
        to elements of the data and labels that make up the dataset.
        """
        self.indices = indices
        self.x_pitches = x_pitches
        self.x_note_lengths = x_note_lengths
        self.y_pitches = y_pitches
        self.y_note_lengths = y_note_lengths

    def __len__(self):
        """
        Returns the length of the dataset.
        """
        return len(self.indices)

    def __getitem__(self, idx):
        """
        Takes an index and returns one sample of data.
        """
        id = self.indices[idx]

        pitches = self.x_pitches[id]
        note_lengths = self.x_note_lengths[id]
        next_pitch = self.y_pitches[id]
        next_note_length = self.y_note_lengths[id]

        return pitches, note_lengths, next_pitch, next_note_length

class LSTM(torch.nn.Module):
    """
    An LSTM for music generation.
    """

    def __init__(self, num_pitches, num_note_lengths, embedding_size, hidden_size):
        """
        Takes in the number of pitches in our vocabulary, the number of note lengths in our vocabulary, the embedding
        size to be used by pitches and note lengths, and the hidden state size for the LSTM. There are two embedding
        layers for pitches and note lengths respectively, which are then concatenated together and fed into the LSTM.
        """
        super(LSTM, self).__init__()

        self.pitch_embedding = torch.nn.Embedding(num_pitches, embedding_size)
        self.note_length_embedding = torch.nn.Embedding(num_note_lengths, embedding_size)

        self.lstm = torch.nn.LSTM(2 * embedding_size, hidden_size, bidirectional=bidirectional, dropout=dropout, num_layers=num_layers, batch_first=True) # The two embeddings are concatenated


        if not bidirectional:
            self.pitch_out = torch.nn.Linear(hidden_size, num_pitches)
            self.note_length_out = torch.nn.Linear(hidden_size, num_note_lengths)

        else:
            self.pitch_out = torch.nn.Linear(2 * hidden_size, num_pitches) # Take in the forward and backward outputs from the LSTM
            self.note_length_out = torch.nn.Linear(2* hidden_size, num_note_lengths)

    def forward(self, pitches, note_lengths):
        """
        Forward propagation, only handles fixed length sequences. The pitches input has shape [batch_size, seq_length,
        num_pitches] and the note_lengths input has shape [batch_size, seq_length, num_note_lengths].
        """
        pitch_embeds = self.pitch_embedding(pitches)
        note_length_embeds = self.note_length_embedding(note_lengths)

        input = torch.cat([pitch_embeds, note_length_embeds], dim=2) # Concatenate the note embeddings
        output, _ = self.lstm(input) # Get model outputs for every input in the sequence

        next_pitch = self.pitch_out(output[:,-1]) # Take the last output and pass it into the linear layer
        next_note_length = self.note_length_out(output[:,-1])

        return torch.softmax(next_pitch, dim=1), torch.softmax(next_note_length, dim=1)

def convert(pitches, note_lengths, id_to_pitch, id_to_duration, save=False):
    """
    Takes in a list of pitches and note lengths represented by their unique integer ID's and the mappings from ID's to
    pitches and note lengths. Returns a Stream object representing this piece of music. If save is true saves the stream
    as a MIDI file.
    """
    midi_stream = stream.Stream()
    midi_stream.append(instrument.Piano()) # The instrument to play the stream in

    for i in range(len(pitches)):
        pitch_pattern = id_to_pitch[pitches[i]] # Convert from the ID to a String representation
        note_length_pattern = id_to_duration[note_lengths[i]]

        if "." in pitch_pattern: # If it is a chord
            pitches_in_chord = pitch_pattern.split(".")
            chord_notes = []

            for i in pitches_in_chord:
                new_note = note.Note(i)
                new_note.duration = duration.Duration(note_length_pattern)
                chord_notes.append(new_note)

            new_chord = chord.Chord(chord_notes)
            midi_stream.append(new_chord)

        elif pitch_pattern == "rest": # If it is a rest
            new_note = note.Rest()
            new_note.duration = duration.Duration(note_length_pattern)
            midi_stream.append(new_note)

        elif pitch_pattern != "START": # If it is a note (ignore START sequences)
            new_note = note.Note(pitch_pattern)
            new_note.duration = duration.Duration(note_length_pattern)
            midi_stream.append(new_note)

    midi_stream = midi_stream.chordify()

    if save:
        timestr = time.strftime("%Y%m%d-%H%M%S")
        midi_stream.write('midi', fp=os.path.join(save_dir, 'generated-' + timestr + '.mid'))

    return midi_stream

def extract_notes(folder):
    """
    Takes in the name of a folder where your original music is stored. Converts each MIDI file into a list of chords and
    extracts the pitch and duration of each one. Returns two list of lists, one for notes and one for durations where
    each sub-list represents a song.
    """
    music_path = os.path.join(data_dir, folder)

    all_pitches = []
    all_note_lengths = []

    for song in tqdm(os.listdir(music_path)): # Loop through each song
        if song == ".DS_Store":
            continue

        path = os.path.join(data_dir, folder, song)
        chords = converter.parse(path).chordify() # Convert each song into a sequential list of chords

        song_pitches = []
        song_note_lengths = []

        song_pitches.extend(["START"] * seq_length) # Add a start sequence to the beginning of every song
        song_note_lengths.extend([0] * seq_length)

        for element in chords.flat:
            if isinstance(element, note.Note): # If it is a single note not a chord
                if element.isRest: # If it is a rest
                    song_pitches.append(str(element.name))
                    song_note_lengths.append(element.duration.quarterLength)

                else:
                    song_pitches.append(str(element.nameWithOctave))
                    song_note_lengths.append(element.duration.quarterLength)

            elif isinstance(element, chord.Chord): # If it is a chord
                song_pitches.append(".".join(p.nameWithOctave for p in element.pitches)) # Separate each of the notes in the chord with a "."
                song_note_lengths.append(element.duration.quarterLength)

        all_pitches.append(song_pitches)
        all_note_lengths.append(song_note_lengths)

    return all_pitches, all_note_lengths

def generate_music(model, max_length, pitches, note_lengths):
    """
    Takes in a trained model and generates a sequence of pitches and note_lengths of length max_length. Initially seeds
    the model with the pitches and note length inputs which each have length seq_length. Returns a list of pitches and
    note lengths represented as their unique integer ID's. Samples pitches and note lengths with probabilities specified
    by the models softmax output.
    """
    output_pitches = []
    output_note_lengths = []

    model.eval()

    while len(output_pitches) < max_length:
        pitch_pred, note_length_pred = model(torch.tensor(pitches).unsqueeze(0), torch.tensor(note_lengths).unsqueeze(0))

        next_pitch = np.random.choice(pitch_pred.shape[1], p=pitch_pred[0].detach().numpy())
        next_note_length = np.random.choice(note_length_pred.shape[1], p=note_length_pred[0].detach().numpy())

        output_pitches.append(next_pitch)
        output_note_lengths.append(next_note_length)

        pitches = pitches[1:] + [next_pitch] # Drop the first pitch in the sequence and add the new pitch to the end
        note_lengths = note_lengths[1:] + [next_note_length]

    return output_pitches, output_note_lengths

def get_counts(matrix):
    """
    Takes in a list of lists as input and counts the number of occurrences of each unique element. Intended for getting
    class weights from the pitches or durations of a song dataset.
    """
    counts = {}

    for i in range(len(matrix)):
        for j in matrix[i]:
            if j not in counts:
                counts[j] = 1

            else:
                counts[j] += 1

    return counts

def get_mapping(matrix):
    """
    Takes in a list of lists as input and generates a dictionary mapping each unique element of the entire matrix to an
    ID, along with the inverse. Intended for making one-hot vector mappings from the pitches or durations of a song
    dataset.
    """
    input_to_id = {}
    id_to_input = {}

    count = 0

    for i in range(len(matrix)):
        for j in matrix[i]:
            if j not in input_to_id:
                input_to_id[j] = count
                id_to_input[count] = j
                count += 1

    return input_to_id, id_to_input

def get_onehot(idx, vector_size):
    """
    Generates a one-hot vector of length vector_size where index idx is set to one.
    """
    onehot = [0] * vector_size
    onehot[idx] = 1

    return onehot

def get_partition(num_samples, train_pct, val_pct, test_pct, seed=None):
    """
    Given a number of samples in your dataset, randomly generates lists of indices for training, validation, and
    testing where there are train_pct * num_samples training samples, etc. Ensure that train_pct + val_pct + test_pct
    is equal to 1. Takes in an optional seed argument.
    """
    if random != None:
        random.seed(seed)

    indices = [i for i in range(num_samples)]
    random.shuffle(indices)

    train_idx = int(train_pct * num_samples)
    val_idx = train_idx + int(val_pct * num_samples)
    test_idx = val_idx + int(test_pct * num_samples)

    return indices[:train_idx], indices[train_idx: val_idx], indices[val_idx:test_idx]

def prepare_sequences(pitches, note_lengths, pitch_to_id, duration_to_id):
    """
    Takes in a list of pitches from  every song and a list of durations for every note. Splits them into chunks of length
    seq_length and maps them to the seq_length + 1'th note and duration. Represents all notes and durations with their
    unique ID's.
    """
    x_pitches = []
    x_note_lengths = []

    y_pitches = []
    y_note_lengths = []

    for i in range(len(pitches)): # Each list in pitches/note_lengths represents a song
        song_pitches = pitches[i]
        song_note_lengths = note_lengths[i]

        for j in range(len(song_pitches) - seq_length): # Ignore last part of song if it isnt divisible by seq_length
            pitch_sequence = song_pitches[j: j + seq_length]
            note_length_sequence = song_note_lengths[j: j + seq_length]
            next_pitch = song_pitches[j + seq_length]
            next_note_length = song_note_lengths[j + seq_length]

            x_pitches.append([pitch_to_id[i] for i in pitch_sequence])
            x_note_lengths.append([duration_to_id[i] for i in note_length_sequence])
            y_pitches.append(get_onehot(pitch_to_id[next_pitch], len(pitch_to_id)))
            y_note_lengths.append(get_onehot(duration_to_id[next_note_length], len(duration_to_id)))

    return torch.tensor(x_pitches), torch.tensor(x_note_lengths), torch.tensor(y_pitches), torch.tensor(y_note_lengths)

def train_model(model, optimizer, loss_1, loss_2, train, val):
    """
    Takes in a model, an optimizer, two distinct loss function (one for pitches and one for note lengths), a training
    dataset, and a validation dataset. Trains the model for the specified number of epochs and saves model weights when
    validation accuracy increases.
    """
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True) # Convert the datasets to batched dataloaders
    val_loader = torch.utils.data.DataLoader(val, batch_size=batch_size, shuffle=True)

    best_acc = 0  # The best accuracy on the validation data
    best_model_weights = model.state_dict() # The weights for the model with the best accuracy

    for e in tqdm(range(epochs)):

        overall_loss = 0 # The models loss this epoch
        train_num_correct = 0 # The number of training samples the model has predicted correctly
        val_num_correct = 0 # The number of validation samples the model has predicted correctly

        model.train() # Train the model on the training dataset

        for pitches, note_lengths, next_pitch, next_note_length in train_loader: # Loop through the training batches
            model.zero_grad()

            pitch_pred, note_length_pred = model(pitches, note_lengths)
            loss =  loss_1(pitch_pred, next_pitch.float()) + loss_2(note_length_pred, next_note_length.float())

            overall_loss += loss.item()

            loss.backward()
            optimizer.step()

            target_pitch = torch.argmax(next_pitch, dim=1) # Convert one-hot target pitch vector to a pitch ID
            target_note_length = torch.argmax(next_note_length, dim=1) # Convert one-hot target note length vector to a note length ID

            top_pitch = torch.argmax(pitch_pred, dim=1) # The most likely next pitch
            top_note_length = torch.argmax(note_length_pred, dim=1) # The most likely next note length

            train_num_correct += torch.sum(target_pitch == top_pitch).item()
            train_num_correct += torch.sum(target_note_length == top_note_length).item()

        train_acc = train_num_correct / (2 * len(train))  # Combine the accuracy on pitch and duration

        model.eval() # Evaluate the model on the validation dataset

        for pitches, note_lengths, next_pitch, next_note_length in val_loader: # Loop through the validation batches
            pitch_pred, note_length_pred = model(pitches, note_lengths)

            target_pitch = torch.argmax(next_pitch, dim=1) # Convert one-hot target pitch vector to a pitch ID
            target_note_length = torch.argmax(next_note_length, dim=1) # Convert one-hot target note length vector to a note length ID

            top_pitch = torch.argmax(pitch_pred, dim=1) # The most likely next pitch
            top_note_length = torch.argmax(note_length_pred, dim=1) # The most likely next note length

            val_num_correct += torch.sum(target_pitch == top_pitch).item()
            val_num_correct += torch.sum(target_note_length == top_note_length).item()

        val_acc = val_num_correct / (2 * len(val)) # Combine the accuracy on pitch and duration

        if val_acc > best_acc: # If we found a new best model
            best_acc = val_acc
            best_model_weights = model.state_dict().copy()

            with open(os.path.join(save_dir, "best_lstm_weights"), "wb") as file:
                pickle.dump(best_model_weights, file)

        print("Epoch: ", e)
        print("Model Loss: ", overall_loss)
        print("Training Accuracy: ", train_acc)
        print("Validation Accuracy: ", val_acc)
        print("Best Accuracy: ", best_acc)

    return best_acc, best_model_weights

with open(os.path.join(save_dir, "notes.pickle"), "rb") as file:
    notes = pickle.load(file)

with open(os.path.join(save_dir, "durations.pickle"), "rb") as file:
    durations = pickle.load(file)

pitch_to_id, id_to_pitch = get_mapping(notes) # Map each pitch to a unique ID
duration_to_id, id_to_duration = get_mapping(durations) # Match each note length to a unique ID

pitch_counts = get_counts(notes) # How many of each type of pitch are there
duration_counts = get_counts(durations) # How many of each note length are there

x_pitches, x_note_lengths, y_pitches, y_note_lengths = prepare_sequences(notes, durations, pitch_to_id, duration_to_id)
train_idx, val_idx, test_idx = get_partition(x_pitches.shape[0], .6, .2, .2, seed=42) # Split the data into train, val, and test sets

train = SongData(train_idx, x_pitches, x_note_lengths, y_pitches, y_note_lengths) # Training dataset
val = SongData(val_idx, x_pitches, x_note_lengths, y_pitches, y_note_lengths) # Validation dataset
test = SongData(test_idx, x_pitches, x_note_lengths, y_pitches, y_note_lengths) # Testing dataset

model = LSTM(len(pitch_to_id), len(duration_to_id), embedding_size, hidden_size)
optimizer = torch.optim.Adam(model.parameters(), lr)
loss_1 = torch.nn.BCELoss()
loss_2 = torch.nn.BCELoss()

# acc, model_weights = train_model(model, optimizer, loss_1, loss_2, train, val)

with open(os.path.join(save_dir, "best_lstm_weights"), "rb") as file:
    weights = pickle.load(file)
    model.load_state_dict(weights)

pitch_seed = [pitch_to_id["START"]] * seq_length
note_length_seed = [duration_to_id[0]] * seq_length

pitches, note_lengths = generate_music(model, 100, pitch_seed, note_length_seed)
song = convert(pitches, note_lengths, id_to_pitch, id_to_duration, save=True)

player = midi.realtime.StreamPlayer(song) # Play the song
player.play()