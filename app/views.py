from app import app
import base64
from flask import request, render_template,send_file,Response,jsonify
from synthesizer.inference import Synthesizer
from encoder import inference as encoder
from vocoder import inference as vocoder
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf
from base64 import b64decode
import uuid
import os
import dialogflow
from google.api_core.exceptions import InvalidArgument
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = 'newagent-skrb-2d1636900f1a.json'
DIALOGFLOW_PROJECT_ID = 'newagent-skrb'
DIALOGFLOW_LANGUAGE_CODE = 'en'
SESSION_ID = 'me'
encoder_weights = Path("encoder/saved_models/pretrained.pt")
vocoder_weights = Path("vocoder/saved_models/pretrained/pretrained.pt")
syn_dir = Path("synthesizer/saved_models/logs-pretrained/taco_pretrained")
encoder.load_model(encoder_weights)
synthesizer = Synthesizer(syn_dir)
vocoder.load_model(vocoder_weights)

@app.route('/generate',methods=["GET","POST"])
def generate():
    text_to_be_analyzed = request.form['text']

    session_client = dialogflow.SessionsClient()
    session = session_client.session_path(DIALOGFLOW_PROJECT_ID, SESSION_ID)
    text_input = dialogflow.types.TextInput(text=text_to_be_analyzed, language_code=DIALOGFLOW_LANGUAGE_CODE)
    query_input = dialogflow.types.QueryInput(text=text_input)
    try:
        response = session_client.detect_intent(session=session, query_input=query_input)
    except InvalidArgument:
        raise
    text=response.query_result.fulfillment_text
    in_fpath = Path("audio.wav")
    reprocessed_wav = encoder.preprocess_wav(in_fpath)
    original_wav, sampling_rate = librosa.load(in_fpath)
    preprocessed_wav = encoder.preprocess_wav(original_wav, sampling_rate)
    embed = encoder.embed_utterance(preprocessed_wav)
    specs = synthesizer.synthesize_spectrograms([text], [embed])
    generated_wav = vocoder.infer_waveform(specs[0])
    generated_wav = np.pad(generated_wav, (0, synthesizer.sample_rate), mode="constant")
    encoded_gen_wav_bytes= base64.b64encode(generated_wav)
    encoded_gen_wav_string = encoded_gen_wav_bytes.decode('utf-8')

    res={
        "data":encoded_gen_wav_string,
        "rate":synthesizer.sample_rate,
        "text":text

    }
    # sf.write("demo_output.wav", generated_wav.astype(np.float32), synthesizer.sample_rate)
    return jsonify(res),200
@app.route('/newVoice',methods=["GET","POST"])
def newVoice():
    try:
        s = request.form['base64']
        b = b64decode(s.split(',')[1])
        sf.write("audio1.wav", b)
        return Response("ok", status=200, mimetype='application/json')
    except Exception as e:
        return Response("ok", status=500, mimetype='application/json')