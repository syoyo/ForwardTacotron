# Load pretrained models
from notebook_utils.synthesize import *
#import IPython.display as ipd
init_hparams('notebook_utils/pretrained_hparams.py')
tts_model = get_forward_model('pretrained/forward_100K.pyt')
#voc_model = get_wavernn_model('pretrained/wave_800K.pyt') 

# Synthesize with normal speed (alpha=1.0)
input_text = 'In the middle of difficulty lies opportunity.'
wav = synthesize(input_text, tts_model, 'griffinlim', alpha=1.0)
print(wav)
#ipd.Audio(wav, rate=hp.sample_rate)
