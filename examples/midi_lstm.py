'''This example demonstrates the use of lstm to train the drum tracks of Metallica 
   and generate new tracks in MIDI format, which can be played by QuickTime or other
   music players.

Based on Keunwoo Choi' Github: https://github.com/keunwoochoi/LSTMetallica.

Please install [python-midi](https://github.com/vishnubob/python-midi) to handle midi file.

References:
- Keunwoo Choi, George Fazekas, Mark Sandler,
  "Text-based LSTM networks for Automatic Music Composition",
  https://arxiv.org/abs/1604.05358

'''

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.utils.data_utils import get_file
import random
import os
import pdb
import midi

# Parameters for MIDI LSTM model
maxlen = 128 # Max input length
num_hidden_units = 10 # Number of hidden units of LSTM
step = 8
num_char_pred = 17 * 30 # Number of predicted char
num_layers = 2
num_epoch = 1
batch_size = 128
diversity = 0.9
result_directory = 'result_midi_lstm/'

# Parameters for Drum Notes
PPQ = 480 # Pulse per quater note
event_per_bar = 16 # to quantise.
min_ppq = PPQ / (event_per_bar / 4)

allowed_pitch = [36, 38, 42, 46, 41, 45, 48, 51, 49]

pitch_to_midipitch = {
	36:midi.C_3,
	38:midi.D_3, 
	39:midi.Eb_3,
	41:midi.F_3,
	42:midi.Gb_3,
	45:midi.A_3,
	46:midi.Bb_3,
	48:midi.C_4,
	49:midi.Db_4,
	51:midi.Eb_4
}

class Note:
	def __init__(self, pitch, c_tick):
		self.pitch = pitch
		self.c_tick = c_tick

	def add_index(self, idx):
		self.idx = idx

class Note_List():
	def __init__(self):
		self.notes = []
		self.quantised = False
		self.max_idx = None

	def add_note(self, note):
		self.notes.append(note)

	def quantise(self, minimum_ppq):
		if not self.quantised:
			for note in self.notes:
				note.c_tick = ((note.c_tick+minimum_ppq/2)/minimum_ppq)* minimum_ppq # quantise
				note.add_index(note.c_tick/minimum_ppq)

			self.max_idx = note.idx
			if (self.max_idx + 1) % event_per_bar != 0:
				self.max_idx += event_per_bar - ((self.max_idx + 1) % event_per_bar) # make sure it has a FULL bar at the end.
			self.quantised = True
		return
	
# Convert the text to notes
def conv_text_to_notes(encoded_drums, note_list=None):
	#0b0000000000 0b10000000 ...  -> corresponding note. 
	if note_list == None:
		note_list = Note_List()

	for word_idx, word in enumerate(encoded_drums):
		c_tick_here = word_idx*min_ppq

		for pitch_idx, pitch in enumerate(allowed_pitch):

			if word[pitch_idx+2] == '1':
				new_note = Note(pitch, c_tick_here)
				note_list.add_note(new_note)
	return note_list

# Convert the text to midi format
def conv_text_to_midi(filename):
	if os.path.exists(filename[:-4]+'.mid'):
		return
	f = open(filename, 'r')
	f.readline() # title
	f.readline() # seed sentence
	sentence = f.readline()
	encoded_drums = sentence.split(' ')

	# find the first BAR
	first_bar_idx = encoded_drums.index('BAR')	
	encoded_drums = encoded_drums[first_bar_idx:]
	try:
		encoded_drums = [ele for ele in encoded_drums if ele not in ['BAR', 'SONG_BEGIN', 'SONG_END', '']]
	except:
		pdb.set_trace()

	# prepare output
	note_list = Note_List()
	pattern = midi.Pattern()
	track = midi.Track()
	PPQ = 220
	min_ppq = PPQ / (event_per_bar / 4)
	track.resolution = PPQ
	pattern.append(track)
	velocity = 84
	duration = min_ppq * 9 / 10 
	note_list = conv_text_to_notes(encoded_drums, note_list=note_list)
	
	max_c_tick = 0 
	not_yet_offed = []
	for note_idx, note in enumerate(note_list.notes[:-1]):
		tick_here = note.c_tick - max_c_tick
		pitch_here = pitch_to_midipitch[note.pitch]
		
		on = midi.NoteOnEvent(tick=tick_here, velocity=velocity, pitch=pitch_here)
		track.append(on)
		max_c_tick = max(max_c_tick, note.c_tick)

		for off_idx, waiting_pitch in enumerate(not_yet_offed):
			if off_idx == 0:
				off = midi.NoteOffEvent(tick=duration, pitch=waiting_pitch)
				max_c_tick = max_c_tick + duration
			else:
				off = midi.NoteOffEvent(tick=0, pitch=waiting_pitch)
			track.append(off)
			not_yet_offed = [] 

	# finalise
	if note_list.notes == []:
		print('No notes in %s' % filename)
		return
		pdb.set_trace()
	note = note_list.notes[-1]
	tick_here = note.c_tick - max_c_tick
	pitch_here = pitch_to_midipitch[note.pitch]
	on = midi.NoteOnEvent(tick=tick_here, velocity=velocity, pitch=pitch_here)
	off = midi.NoteOffEvent(tick=duration, pitch=pitch_here)

	for off_idx, waiting_pitch in enumerate(not_yet_offed):
		off = midi.NoteOffEvent(tick=0, pitch=waiting_pitch)

	# end of track event
	eot = midi.EndOfTrackEvent(tick=1)
	track.append(eot)
	midi.write_midifile(filename[:-4]+'.mid', pattern)

# Save the predictions to text file
def save_to_text(path_text):
	f_write = open((path_text), 'w')
	f_write.write('diversity:%4.2f\n' % diversity)

	generated = []
	start_index = random.randint(0, len(text) - maxlen - 1)
	sentence = text[start_index: start_index + maxlen]
	seed_sentence = text[start_index: start_index + maxlen]
	generated = generated + sentence

	for i in xrange(num_char_pred):
		x = np.zeros((1, maxlen, num_chars))	
		for t, char in enumerate(sentence):
			x[0, t, char_indices[char]] = 1.

		preds = model.predict(x, verbose=0)[0]
		next_index = sample(preds, diversity)
		next_char = indices_char[next_index]

		generated.append(next_char)
		# Update sentence
		sentence = sentence[1:]
		sentence.append(next_char)

	f_write.write(' '.join(seed_sentence) + '\n')
	f_write.write(' ' .join(generated))
	f_write.close()

# Sample the predictions for diversity
def sample(index, diversity=1.0):
	index = np.log(index) / diversity
	index = np.exp(index) / np.sum(np.exp(index))
	return np.argmax(np.random.multinomial(1, index, 1))

if not os.path.exists(result_directory):
	os.mkdir(result_directory)

try:
    path = get_file('metallica_drums_text.txt', origin='https://s3.amazonaws.com/datasetpandevirus/metallica_drums_text.txt')
except:
    print('Error downloading dataset, please download it manually:\n'
          '$ wget https://s3.amazonaws.com/datasetpandevirus/metallica_drums_text.txt\n')
text = open(path).read()

print('Building char vocabulary dictionary...')
chord_seq = text.split(' ')
chars = set(chord_seq)
text = chord_seq
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))
num_chars = len(char_indices)

sentences = []
next_chars = []

for i in range(0, len(text) - maxlen, step):
	sentences.append(text[i: i + maxlen])
	next_chars.append(text[i + maxlen])

X = np.zeros((len(sentences), maxlen, num_chars), dtype=np.bool)
y = np.zeros((len(sentences), num_chars), dtype=np.bool)
for i, sentence in enumerate(sentences):
	for t, char in enumerate(sentence):
		X[i, t, char_indices[char]] = 1
	y[i, char_indices[next_chars[i]]] = 1

print('Building model...')
model = Sequential()
for layer_idx in range(num_layers):
	if layer_idx == 0:
		model.add(LSTM(num_hidden_units, return_sequences=True, input_shape=(maxlen, num_chars)))
	else:
		model.add(LSTM(num_hidden_units, return_sequences=False))
	model.add(Dropout(0.2))

model.add(Dense(num_chars))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')

print('Training...')
result = model.fit(X, y, batch_size=batch_size, nb_epoch=num_epoch)

print('Generating new tracks...')
path_text = '%sresult.txt' % (result_directory)
save_to_text(path_text)
conv_text_to_midi(path_text)