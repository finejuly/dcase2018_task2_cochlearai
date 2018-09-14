import h5py
import librosa
import os
import numpy as np
import csv

label=["Acoustic_guitar", "Applause", "Bark", "Bass_drum", "Burping_or_eructation", "Bus", "Cello", "Chime", "Clarinet", "Computer_keyboard", "Cough", "Cowbell", "Double_bass", "Drawer_open_or_close", "Electric_piano", "Fart", "Finger_snapping", "Fireworks", "Flute", "Glockenspiel", "Gong", "Gunshot_or_gunfire", "Harmonica", "Hi-hat", "Keys_jangling", "Knock", "Laughter", "Meow", "Microwave_oven", "Oboe", "Saxophone", "Scissors", "Shatter", "Snare_drum", "Squeak", "Tambourine", "Tearing", "Telephone", "Trumpet", "Violin_or_fiddle", "Writing"]

def make_h5(path, sr=32000):

	filelist = os.listdir(path)
	y_all = []

	if not os.path.exists('data'):
		os.makedirs('data')

	h = h5py.File('./data/data_test_%s.h5'%sr, 'w')
	g = h.create_group('data')

	for f in filelist:
		ext=os.path.splitext(f)[-1]
		if ext=='.wav':
			path_f = os.path.join(path,f)
			y,sr_new = librosa.load(path_f,sr=sr)
			y = y.astype('float16')
			g.create_dataset(f,data=y)


def load_test_data(sr=32000):
	h_data=h5py.File('./data/data_test_%s.h5'%sr,'r') 	
	y = []
	for key in h_data['data'].keys():
		x = np.asarray(h_data['data'][key])
		y.append(x)
	return y

def write_csv(pred, filename):
	h_data=h5py.File('./data/data_test_32000.h5','r') 
	
	with open(filename, 'w') as csvfile:
		fieldnames = ['fname', 'label']
		writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
		writer.writeheader()
	
		k = 0
		for key in h_data['data'].keys():
			x = pred[k,:]
			x_rank = np.argsort(x)
			p1 = label[x_rank[-1]]
			p2 = label[x_rank[-2]]
			p3 = label[x_rank[-3]]
			writer.writerow({'fname': key, 'label': '%s %s %s'%(p1,p2,p3)})
			k += 1