from app import app
import base64
from flask_api import status
from flask import request, render_template,send_file
from synthesizer.inference import Synthesizer
from encoder import inference as encoder
from vocoder import inference as vocoder
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf
from base64 import b64decode
import uuid
encoder_weights = Path("encoder/saved_models/pretrained.pt")
vocoder_weights = Path("vocoder/saved_models/pretrained/pretrained.pt")
syn_dir = Path("synthesizer/saved_models/logs-pretrained/taco_pretrained")
encoder.load_model(encoder_weights)
synthesizer = Synthesizer(syn_dir)
vocoder.load_model(vocoder_weights)

@app.route('/generate',methods=["GET","POST"])
def generate():
    text = "Tonight, I am asking you to believe in Joe and Kamala’s ability to lead this country"
    in_fpath = Path("audio.wav")
    reprocessed_wav = encoder.preprocess_wav(in_fpath)
    original_wav, sampling_rate = librosa.load(in_fpath)
    preprocessed_wav = encoder.preprocess_wav(original_wav, sampling_rate)
    embed = encoder.embed_utterance(preprocessed_wav)
    specs = synthesizer.synthesize_spectrograms([text], [embed])
    generated_wav = vocoder.infer_waveform(specs[0])
    generated_wav = np.pad(generated_wav, (0, synthesizer.sample_rate), mode="constant")
    encoded_gen_wav= base64.b64encode(generated_wav)
    res={
        "data":encoded_gen_wav,
        "rate":synthesizer.sample_rate
    }
    # sf.write("demo_output.wav", generated_wav.astype(np.float32), synthesizer.sample_rate)
    return res
@app.route('/newVoice',methods=["GET","POST"])
def newVoice():
    try:
        s = request.form['base64']
        b = b64decode(s.split(',')[1])
        sf.write("audio.wav", generated_wav.astype(np.float32))
        return status.HTTP_200_OK
    except Exception as e:
        return status.HTTP_500_INTERNAL_SERVER_ERROR