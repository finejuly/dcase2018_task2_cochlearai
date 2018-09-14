from utils import util, model
import glob

# First, put test audio samples in 'audio_test/'.

# input audio is distinguished according to the sample rate.
sample_rate_list=[16000, 32000, 44000]
# sample_rate_list=[32000]

n=0
for sample_rate in sample_rate_list:
	# resample audio and put them into a single h5 file. 
	util.make_h5('./audio_test', sr=sample_rate)

	#load test data from h5 file.
	X_test = util.load_test_data(sample_rate)

	# load trained 5 models' path (same models but 5-fold cross validated)
	model_list=glob.glob('saved_models/%s/**/*.h5'%sample_rate)

	for model_name in model_list:
		#load each model
		m = model.load_model(model_name)
		# model predict
		pred = model.predict(m, X_test, n_class=41)

		# ensemble with geometric mean
		if n == 0:
			total_pred=pred
		else:
			total_pred*=pred

		n+=1

result = total_pred**(1/float(n))
#######################################

# save submission by the format for MAP@3 evaluation.
util.write_csv(result, './submission.csv')
