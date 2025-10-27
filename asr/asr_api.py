import torch
import torchaudio
import tempfile
from flask import Flask, request, jsonify
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

app = Flask(__name__)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load model and processor once at startup
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")
model = model.to(device)

@app.route('/ping', methods=['GET'])
def ping():
    return 'pong'

@app.route('/asr', methods=['POST'])
def asr():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    # Do some validation checks to ensure the file is valid
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Assume that only .mp3 files will be processed
    if not file.filename.lower().endswith('.mp3'):
        return jsonify({'error': 'File must be MP3 format'}), 400
    
    try:

        with tempfile.NamedTemporaryFile(delete=True, suffix='.mp3') as temp_file:
            file.save(temp_file.name)

            # Load audio with torchaudio
            waveform, sample_rate = torchaudio.load(temp_file.name)

            # Calculate duration before any transformations
            duration = waveform.shape[1] / sample_rate

            # Resample to 16kHz if necessary
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
                waveform = resampler(waveform)
                sample_rate = 16000

            # Convert to 1D mono waveform from stereo if necessary
            # torchaudio returns shape [channels, time]
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0)  # convert to mono
            else:
                waveform = waveform.squeeze(0)

            # Extract features
            inputs = processor(waveform, sampling_rate=16000, return_tensors="pt", padding=True)

            # Move inputs to GPU
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                logits = model(**inputs).logits

            # Decode prediction to text
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = processor.batch_decode(predicted_ids)[0]
            transcription = ''.join(transcription)

            return jsonify({
                'transcription': transcription,
                'duration': str(duration)
            })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8001, debug=True)