# dcase2018_task2_cochlearai
Cochlear.ai submission for dcase2018 task2

You can download full data including models in the following links:
https://console.cloud.google.com/storage/browser/cochlearai_public/dcase2018_task2_cochlearai/


# How to run

1. Download the models from the above links. You can find them in '/saved_models/'.
2. Put your audio files as 'audio_test/a.wav'.
3. Modify  the list of sample_rate_list in main_pred.py, as [16000], [32000], [44100], or combination of them as [16000, 44100].
(Note that those sample rates corresponds to the models, not the data. Audio files will be resampled for each sample rate models.)
4. Run main_pred.py
5. The results will be saved as submission.csv.


# requirements:

h5py, librosa, keras, tensorflow, kapre
