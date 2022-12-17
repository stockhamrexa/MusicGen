import music21
import os
import pickle
import pypianoroll as pp
import torch

from tqdm import tqdm

# Constants
data_dir = "data/GAN/" # Where is your data located
save_dir = "data/GAN/saved_data" # Where should your outputs and model weights be saved

batch_size = 16 # Size of batches
epochs = 50 # Epochs of training
generator_lr = .001 # The generators learning rate
discriminator_lr = .0001 # The discriminators learning rate
real_label = 1 # Label for data from the real dataset
fake_label = 0 # Label for data from the generator

drum_track = 0 # Which of the 5 tracks is the drums in every song
n_bars = 4 # The number of bars in each generated phrase of music
n_pitches = 128
n_steps_per_bar = 24 # The number of timesteps for a single beat (4 beats to a bar)
n_tracks = 5 # The number of instruments

# A batch of latent noise we can use to track the progress of the discriminator
torch.manual_seed(0)
fixed_chords_noise = torch.randn(batch_size, n_bars * n_steps_per_bar, 1, 1)
fixed_style_noise = torch.randn(batch_size, n_steps_per_bar)
fixed_melody_noise = torch.randn(batch_size, n_bars * n_steps_per_bar, n_tracks, 1)
fixed_groove_noise = torch.randn(batch_size, n_steps_per_bar, n_tracks)

class SongData(torch.utils.data.Dataset):
    """
    A dataset class for phrases of songs.
    """

    def __init__(self, phrases):
        """
        Takes in a list of song phrases (real data).
        """
        self.phrases = phrases

    def __len__(self):
        """
        Returns the length of the dataset.
        """
        return len(self.phrases)

    def __getitem__(self, idx):
        """
        Takes an index and returns one sample of data.
        """
        return self.phrases[idx]

class Generator(torch.nn.Module):
    """
    A generator for the MuseGAN used to create polyphonic multi-track music.
    """

    def __init__(self):
        """
        Initializes the MuseGAN generator.
        """
        super(Generator, self).__init__()

        # All of the TemporalNetwork objects used by the MuseGAN to create the input to the generators
        self.chords_network = TemporalNetwork()
        self.drum_network = TemporalNetwork()
        self.piano_network = TemporalNetwork()
        self.guitar_network = TemporalNetwork()
        self.bass_network = TemporalNetwork()
        self.strings_network = TemporalNetwork()

        # All of the BarGenerator objects used by the MuseGAN Generator to create music for each track
        self.drum_generator = BarGenerator()
        self.piano_generator = BarGenerator()
        self.guitar_generator = BarGenerator()
        self.bass_generator = BarGenerator()
        self.strings_generator = BarGenerator()

    def forward(self, chords_noise, style_noise, melody_noise, groove_noise):
        """
        Performs forward propagation. Takes in four noise vectors where chords noise has shape [batch_size,
        n_bars * n_steps_per_bar, 1, 1], style_noise has shape [batch_size, n_steps_per_bar], melody_noise has size
        [batch_size, n_bars * n_steps_per_bar, n_tracks, 1], and groove_noise has size [batch_size, n_steps_per_bar,
        n_tracks].
        """
        song = None

        chords = self.chords_network(chords_noise)
        drums = self.drum_network(melody_noise[:, :, 0].unsqueeze(dim=3))
        piano = self.piano_network(melody_noise[:, :, 3].unsqueeze(dim=3))
        guitar = self.guitar_network(melody_noise[:, :, 1].unsqueeze(dim=3))
        bass = self.bass_network(melody_noise[:, :, 2].unsqueeze(dim=3))
        strings = self.strings_network(melody_noise[:, :, 4].unsqueeze(dim=3))

        for i in range(n_bars):
            bar_input = torch.cat((chords[:, :, i], style_noise), dim=1)

            drum_input = torch.cat((bar_input, drums[:, :, i], groove_noise[:, :, 0]), dim=1) # Size [batch_size, n_bars * n_steps_per_bar]
            drum_track = self.drum_generator(drum_input)

            piano_input = torch.cat((bar_input, piano[:, :, i], groove_noise[:, :, 1]), dim=1)
            piano_track = self.piano_generator(piano_input)

            guitar_input = torch.cat((bar_input, guitar[:, :, i], groove_noise[:, :, 2]), dim=1)
            guitar_track = self.guitar_generator(guitar_input)

            bass_input = torch.cat((bar_input, bass[:, :, i], groove_noise[:, :, 3]), dim=1)
            bass_track = self.bass_generator(bass_input)

            strings_input = torch.cat((bar_input, strings[:, :, i], groove_noise[:, :, 4]), dim=1)
            strings_track = self.strings_generator(strings_input)

            full_bar = torch.cat((drum_track, piano_track, guitar_track, bass_track, strings_track), dim=3) # A bar of music with n_tracks of size [n_batches, n_steps_per_bar, n_pitches, 5]

            if i == 0: # If it is the first bar
                song = full_bar

            else:
                song = torch.cat((song, full_bar), dim=1)

        return song

class BarGenerator(torch.nn.Module):
    """
    A neural network used to generate a single bar of music for one instrument (track).
    """

    def __init__(self):
        """
        Initializes the bar generator.
        """
        super(BarGenerator, self).__init__()
        self.linear = torch.nn.Linear(n_bars * n_steps_per_bar, 1024)
        self.batch_norm = torch.nn.BatchNorm1d(num_features=1024, momentum=.9)

        self.conv1 = torch.nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=(2, 1), stride=(2, 1))
        self.batch_norm1 = torch.nn.BatchNorm2d(num_features=512, momentum=.9)

        self.conv2 = torch.nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=(2, 1), stride=(2, 1))
        self.batch_norm2 = torch.nn.BatchNorm2d(num_features=256, momentum=.9)

        self.conv3 = torch.nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=(2, 1), stride=(2, 1))
        self.batch_norm3 = torch.nn.BatchNorm2d(num_features=256, momentum=.9)

        self.conv4 = torch.nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=(2, 1), stride=(2, 1))
        self.batch_norm4 = torch.nn.BatchNorm2d(num_features=256, momentum=.9)

        self.conv5 = torch.nn.ConvTranspose2d(in_channels=256, out_channels=n_steps_per_bar, kernel_size=(4, 1), stride=(4, 1))

    def forward(self, input):
        """
        Forward propagation. Takes in a single input vector of size [batch_size, n_bars * n_steps_per_bar]. Returns a
        vector of size [batch_size, n_steps_per_bar, n_pitches, 1].
        """
        x = self.linear(input)
        x = self.batch_norm(x)
        x = torch.nn.functional.relu(x)
        x = x.reshape((input.shape[0], 512, 2, 1))
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = torch.nn.functional.relu(x)
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = torch.nn.functional.relu(x)
        x = self.conv3(x)
        x = self.batch_norm3(x)
        x = torch.nn.functional.relu(x)
        x = self.conv4(x)
        x = self.batch_norm4(x)
        x = torch.nn.functional.relu(x)
        x = self.conv5(x)
        x = torch.sigmoid(x)

        return x

class TemporalNetwork(torch.nn.Module):
    """
    A neural network that consists of convolutional transpose layers. It takes in a noise vector of size
    [batch_size, n_bars * n_steps_per_bar, 1, 1] and produces an output of shape [batch_size, n_steps_per_bar, n_bars].
    """

    def __init__(self):
        """
        Initializes the temporal network.
        """
        super(TemporalNetwork, self).__init__()

        self.conv1 = torch.nn.ConvTranspose2d(in_channels=n_bars * n_steps_per_bar, out_channels=1024, kernel_size=(2, 1), stride=(2, 1))
        self.batch_norm1 = torch.nn.BatchNorm2d(num_features=1024, momentum=.9)

        self.conv2 = torch.nn.ConvTranspose2d(in_channels=1024, out_channels=n_steps_per_bar, kernel_size=(n_bars - 1, 1), stride=(1, 1))
        self.batch_norm2 = torch.nn.BatchNorm2d(num_features=n_steps_per_bar, momentum=.9)

    def forward(self, noise):
        """
        Forward propagation.
        """
        x = self.conv1(noise)
        x = self.batch_norm1(x)
        x = torch.nn.functional.relu(x)
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = torch.nn.functional.relu(x)

        return x.squeeze()

class Discriminator(torch.nn.Module):
    """
    A discriminator for the MuseGAN used to differentiate between real and artificially generated multi-track music.
    """

    def __init__(self):
        """
        Initializes the MuseGAN discriminator.
        """
        super(Discriminator, self).__init__()

        self.conv1 = torch.nn.Conv2d(in_channels=n_bars * n_steps_per_bar, out_channels=256, kernel_size=1, stride=1, bias=True)

        self.conv2 = torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, bias=True)
        self.batch_norm2 = torch.nn.BatchNorm2d(num_features=512, momentum=.9)

        self.conv3 = torch.nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=2, stride=2, bias=True)
        self.batch_norm3 = torch.nn.BatchNorm2d(num_features=1024, momentum=.9)

        self.conv4 = torch.nn.Conv2d(in_channels=1024, out_channels=64, kernel_size=1, stride=1, bias=True)
        self.linear = torch.nn.Linear(4032, 1)

    def forward(self, input):
        """
        Forward propagation. Takes a batch of songs as input in the shape [batch_size, n_bars * n_steps_per_bar,
        n_pitches, n_tracks].
        """
        x = self.conv1(input)
        x = torch.nn.functional.leaky_relu(x, negative_slope=.2, inplace=True)
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = torch.nn.functional.leaky_relu(x, negative_slope=.2, inplace=True)
        x = self.conv3(x)
        x = self.batch_norm3(x)
        x = torch.nn.functional.leaky_relu(x, negative_slope=.2, inplace=True)
        x = self.conv4(x)
        x = torch.flatten(x, start_dim=1, end_dim=3)
        x = self.linear(x)

        return torch.sigmoid(x)

def extract_phrases(num_songs=5000, save=False):
    """
    Loops through num_songs of the songs in our "GAN/original_music" folder and splits them into phrases of length n_bars *
    n_steps_per_bar. There are 5000 songs in the folder. If save is set to True, saves a copy of the phrases list.
    """
    phrases = []
    files_traversed = 0

    for file in os.listdir(os.path.join(data_dir, "original_music")):
        if file[-4:] == ".npz": # Only use .npz files
            files_traversed += 1

            song = pp.load(os.path.join(data_dir, "original_music", file))
            song.binarize(threshold=0) # Remove note intensity to make each note a one-hot vector representation
            song = song.get_stacked_pianoroll() # Convert the song into pianoroll format
            song = song.astype(int) # Convert from boolean to int

            for i in range(0, song.shape[0] - n_bars * n_steps_per_bar, n_bars * n_steps_per_bar):
                phrase = song[i:i + n_bars * n_steps_per_bar]
                phrases.append(phrase)

        if files_traversed > num_songs:
            break

    if save:
        with open(os.path.join(save_dir, "phrases.pickle"), "wb") as file:
            pickle.dump(phrases, file)

    return phrases

def convert_music(song, epoch):
    """
    Takes in a song generated by the MuseGAN and converts it to a pypianoroll object. The pypianoroll object is then
    saved.
    """
    if type(song) == torch.Tensor:
        song = song.cpu().detach().numpy()

    if len(song.shape) > 3: # If it was a batch of songs
        for i in range(song.shape[0]):
            phrase = pp.Multitrack()

            for j in range(n_tracks):
                track_roll = song[i, :, :, j] # The pianoroll for each of the n_tracks tracks

                if j == 0:  # Drums
                    track = pp.Track(track_roll, 0, True, "Drums")
                    phrase.append_track(track, track_roll, 0, True, "Drums")

                elif j == 1:  # Piano
                    track = pp.Track(track_roll, 0, False, "Piano")
                    phrase.append_track(track, track_roll, 0, False, "Piano")

                elif j == 2:  # Guitar
                    track = pp.Track(track_roll, 24, False, "Guitar")
                    phrase.append_track(track, track_roll, 24, False, "Guitar")

                elif j == 3:  # Bass
                    track = pp.Track(track_roll, 32, False, "Bass")
                    phrase.append_track(track, track_roll, 32, False, "Bass")

                else:  # Strings
                    track = pp.Track(track_roll, 48, False, "Strings")
                    phrase.append_track(track, track_roll, 48, False, "Strings")

            phrase.binarize(threshold=.95)
            phrase.write(os.path.join(save_dir, "fake-" + str(i) + "-" + str(epoch) + ".mid"))

    else:
        phrase = pp.Multitrack()

        for j in range(n_tracks):
            track_roll = song[:, :, j] # The pianoroll for each of the n_tracks tracks

            if j == 0: #Drums
                track = pp.Track(track_roll, 0, True, "Drums")
                phrase.append_track(track, track_roll, 0, True, "Drums")

            elif j == 1: # Piano
                track = pp.Track(track_roll, 0, False, "Piano")
                phrase.append_track(track, track_roll, 0, False, "Piano")

            elif j == 2: # Guitar
                track = pp.Track(track_roll, 24, False, "Guitar")
                phrase.append_track(track, track_roll, 24, False, "Guitar")

            elif j == 3: # Bass
                track = pp.Track(track_roll, 32, False, "Bass")
                phrase.append_track(track, track_roll, 32, False, "Bass")

            else: # Strings
                track = pp.Track(track_roll, 48, False, "Strings")
                phrase.append_track(track, track_roll, 48, False, "Strings")

        phrase.binarize(threshold=.95)
        phrase.write(os.path.join(save_dir, "fake-" + str(epoch) + ".mid"))

def play_songs(song_list):
    """
    Given a list of songs represented as pypianoroll objects, play each of them one after another.
    """
    for song in song_list:
        song.write(os.path.join(save_dir, "temp.mid"))
        stream = music21.midi.translate.midiFilePathToStream(os.path.join(save_dir, "temp.mid"))
        player = music21.midi.realtime.StreamPlayer(stream)  # Play the song
        player.play()
        os.remove(os.path.join(save_dir, "temp.mid"))

def train_model(generator, discriminator, criterion, optimizerD, optimizerG, dataloader):
    """
    Trains the MuseGAN, saving the progress at every epoch. Takes in the generator and discriminator models, the loss
    function, an optimizer for both the generator and discriminator respectively, and a dataloader containing all of
    the real phrases.
    """
    D_accuracies = [] # Discriminator accuracies each epoch
    G_accuracies = [] # Generator accuracies each epoch

    D_losses = [] # Discriminator losses each epoch
    G_losses = [] # Generator losses each epoch

    D_real_output = []  # The average output of the discriminator on real data each epoch
    D_fake_output = []  # The average output of the discriminator on fake data each epoch

    for epoch in tqdm(range(epochs)):
        print("Epoch: ", epoch)

        discriminator.train()
        generator.train()

        D_accuracy = 0
        G_accuracy = 0

        D_loss = 0
        G_loss = 0

        real_output = 0
        fake_output = 0

        for i, data in enumerate(dataloader): # Loop through all of the real data
            discriminator.zero_grad()
            labels = torch.full((data.shape[0], 1), real_label)
            output = discriminator(data.float())
            errD_real = criterion(output, labels)  # The discriminators error on the real data
            errD_real.backward()

            real_output += torch.mean(output).item()
            D_accuracy += torch.sum((output > .5) == (labels > .5)).item()

            # Noise for the generator, where data.shape[0] is the batch size
            chords_noise = torch.randn(data.shape[0], n_bars * n_steps_per_bar, 1, 1)
            style_noise = torch.randn(data.shape[0], n_steps_per_bar)
            melody_noise = torch.randn(data.shape[0], n_bars * n_steps_per_bar, n_tracks, 1)
            groove_noise = torch.randn(data.shape[0], n_steps_per_bar, n_tracks)

            labels = torch.full((data.shape[0], 1), fake_label)
            fake = generator(chords_noise, style_noise, melody_noise, groove_noise)
            output = discriminator(fake.detach())
            errD_fake = criterion(output, labels)  # The discriminators error on the real data
            errD_fake.backward()

            optimizerD.step()

            fake_output += torch.mean(output).item()
            D_accuracy += torch.sum((output < .5) == (labels < .5)).item()
            G_accuracy += torch.sum((output > .5) == ((1 - labels) > .5)).item()

            errD = errD_real + errD_fake
            D_loss += errD.item()

            generator.zero_grad()
            labels = torch.full((data.shape[0], 1), real_label)
            output = discriminator(fake)
            errG = criterion(output, labels)
            errG.backward(retain_graph=True)
            optimizerG.step()

            G_loss += errG.item()

            if i == len(dataloader) - 1:  # If it is the last iteration, sample the model and save weights
                with open(os.path.join(save_dir, "D_Weights-" + str(epoch) + ".pickle"), "wb") as file:
                    pickle.dump(discriminator.state_dict().copy(), file)

                with open(os.path.join(save_dir, "G_Weights-" + str(epoch) + ".pickle"), "wb") as file:
                    pickle.dump(generator.state_dict().copy(), file)

                new_songs = generator(fixed_chords_noise, fixed_style_noise, fixed_melody_noise, fixed_groove_noise)
                convert_music(new_songs, epoch)

                with open(os.path.join(save_dir, "songs-" + str(epoch) + ".pickle"), "wb") as file:
                    pickle.dump(new_songs, file)

        D_accuracies.append(D_accuracy / (2 * (batch_size * len(dataloader))))
        G_accuracies.append(G_accuracy / (batch_size * len(dataloader)))

        D_losses.append(D_loss / len(dataloader))
        G_losses.append(G_loss / len(dataloader))

        D_real_output.append(real_output / len(dataloader))
        D_fake_output.append(fake_output / len(dataloader))

        print("D Accuracy: ", D_accuracies[-1], "G Accuracy: ", G_accuracies[-1], "D Loss: ", D_losses[-1], "G Loss: ", G_losses[-1], "Real Mean: ", D_real_output[-1], "Fake Mean: ", D_fake_output[-1])

    return D_losses, G_losses

phrases = extract_phrases(num_songs=15, save=True) # Extract phrases from the music

with open(os.path.join(save_dir, "phrases.pickle"), "rb") as file:
    phrases = pickle.load(file)

real_data = SongData(phrases) # Convert the phrases into a dataset

generator = Generator() # The MuseGAN generator
discriminator = Discriminator() # The MuseGAN discriminator

criterion = torch.nn.BCELoss() # Our loss function
optimizerD = torch.optim.Adam(list(discriminator.parameters()), lr=discriminator_lr, betas=(.5, 0.999)) # The optimizer for the discriminator
optimizerG = torch.optim.Adam(list(generator.parameters()), lr=discriminator_lr, betas=(.5, 0.999)) # The optimizer for the generator
dataloader = torch.utils.data.DataLoader(real_data, batch_size=batch_size, shuffle=True)  # Convert the datasets to batched dataloaders

D_losses, G_losses = train_model(generator, discriminator, criterion, optimizerD, optimizerG, dataloader)