"""
Flask Web App for DTLN Multi-Method Audio Processing
Supports: Deterministic, Stochastic, and Traditional methods
Author: Generated for DTLN Evaluation
"""

from flask import Flask, render_template, request, jsonify, send_file, url_for
import os
import uuid
import numpy as np
import soundfile as sf
import onnxruntime
from werkzeug.utils import secure_filename
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal
from pystoi import stoi
from pesq import pesq
import traceback

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max
app.config['ALLOWED_EXTENSIONS'] = {'wav', 'flac', 'mp3'}

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
os.makedirs('static/spectrograms', exist_ok=True)

# DTLN Model Configuration
BLOCK_LEN = 512
BLOCK_SHIFT = 128
SAMPLE_RATE = 16000

# Initialize ONNX models (will be loaded on first use)
interpreter_1 = None
interpreter_2 = None
model_inputs_1 = None
model_inputs_2 = None
model_input_names_1 = None
model_input_names_2 = None


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def load_onnx_models():
    """Load ONNX models for DTLN inference"""
    global interpreter_1, interpreter_2, model_inputs_1, model_inputs_2
    global model_input_names_1, model_input_names_2
    
    if interpreter_1 is not None:
        return
    
    # Load model 1
    model_1_path = os.path.join('pretrained_model', 'model_1.onnx')
    model_2_path = os.path.join('pretrained_model', 'model_2.onnx')
    
    if not os.path.exists(model_1_path) or not os.path.exists(model_2_path):
        raise FileNotFoundError("ONNX models not found. Please place model_1.onnx and model_2.onnx in the pretrained_model directory.")
    
    interpreter_1 = onnxruntime.InferenceSession(model_1_path)
    model_input_names_1 = [inp.name for inp in interpreter_1.get_inputs()]
    model_inputs_1 = {
        inp.name: np.zeros(
            [dim if isinstance(dim, int) else 1 for dim in inp.shape],
            dtype=np.float32)
        for inp in interpreter_1.get_inputs()
    }
    
    interpreter_2 = onnxruntime.InferenceSession(model_2_path)
    model_input_names_2 = [inp.name for inp in interpreter_2.get_inputs()]
    model_inputs_2 = {
        inp.name: np.zeros(
            [dim if isinstance(dim, int) else 1 for dim in inp.shape],
            dtype=np.float32)
        for inp in interpreter_2.get_inputs()
    }


def load_audio(filepath, target_sr=16000):
    """Load audio file and resample if necessary"""
    audio, fs = sf.read(filepath)
    
    # Convert to mono if stereo
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
    
    # Resample if necessary
    if fs != target_sr:
        from scipy.signal import resample
        num_samples = int(len(audio) * target_sr / fs)
        audio = resample(audio, num_samples)
    
    return audio, target_sr


def add_gaussian_noise(audio, snr_db):
    """Add Gaussian random noise to audio"""
    audio_power = np.mean(audio ** 2)
    noise_power = audio_power / (10 ** (snr_db / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), len(audio))
    return audio + noise, noise


def add_white_noise(audio, snr_db):
    """Add white noise to audio"""
    audio_power = np.mean(audio ** 2)
    noise_power = audio_power / (10 ** (snr_db / 10))
    noise = np.random.uniform(-1, 1, len(audio)) * np.sqrt(noise_power * 3)
    return audio + noise, noise


def mix_audio_with_snr(clean, noise, snr_db):
    """Mix clean audio with noise at specified SNR"""
    # Ensure same length
    if len(noise) < len(clean):
        # Repeat noise if shorter
        repeats = int(np.ceil(len(clean) / len(noise)))
        noise = np.tile(noise, repeats)[:len(clean)]
    else:
        # Truncate noise if longer
        noise = noise[:len(clean)]
    
    # Calculate scaling factor
    clean_power = np.mean(clean ** 2)
    noise_power = np.mean(noise ** 2)
    
    if noise_power == 0:
        noise_power = 1e-10
    
    scale = np.sqrt(clean_power / (noise_power * (10 ** (snr_db / 10))))
    scaled_noise = noise * scale
    
    mixed = clean + scaled_noise
    
    # Normalize to prevent clipping
    max_val = np.max(np.abs(mixed))
    if max_val > 1.0:
        mixed = mixed / max_val
        scaled_noise = scaled_noise / max_val
    
    return mixed, scaled_noise


def process_dtln(audio_input, fs=16000):
    """Process audio using DTLN ONNX models"""
    load_onnx_models()
    
    if fs != SAMPLE_RATE:
        raise ValueError(f'This model only supports {SAMPLE_RATE}Hz sampling rate.')
    
    # Preallocate output
    out_file = np.zeros((len(audio_input)))
    
    # Create buffers
    in_buffer = np.zeros((BLOCK_LEN)).astype('float32')
    out_buffer = np.zeros((BLOCK_LEN)).astype('float32')
    
    # Reset model states
    for key in model_inputs_1:
        model_inputs_1[key] = np.zeros_like(model_inputs_1[key])
    for key in model_inputs_2:
        model_inputs_2[key] = np.zeros_like(model_inputs_2[key])
    
    # Calculate number of blocks
    num_blocks = (audio_input.shape[0] - (BLOCK_LEN - BLOCK_SHIFT)) // BLOCK_SHIFT
    
    # Process blocks
    for idx in range(num_blocks):
        # Shift values and write to buffer
        in_buffer[:-BLOCK_SHIFT] = in_buffer[BLOCK_SHIFT:]
        in_buffer[-BLOCK_SHIFT:] = audio_input[idx*BLOCK_SHIFT:(idx*BLOCK_SHIFT)+BLOCK_SHIFT]
        
        # Calculate FFT
        in_block_fft = np.fft.rfft(in_buffer)
        in_mag = np.abs(in_block_fft)
        in_phase = np.angle(in_block_fft)
        
        # Reshape for model input
        in_mag = np.reshape(in_mag, (1, 1, -1)).astype('float32')
        
        # First model
        model_inputs_1[model_input_names_1[0]] = in_mag
        model_outputs_1 = interpreter_1.run(None, model_inputs_1)
        out_mask = model_outputs_1[0]
        model_inputs_1[model_input_names_1[1]] = model_outputs_1[1]
        
        # IFFT
        estimated_complex = in_mag * out_mask * np.exp(1j * in_phase)
        estimated_block = np.fft.irfft(estimated_complex)
        
        # Reshape for second model
        estimated_block = np.reshape(estimated_block, (1, 1, -1)).astype('float32')
        
        # Second model
        model_inputs_2[model_input_names_2[0]] = estimated_block
        model_outputs_2 = interpreter_2.run(None, model_inputs_2)
        out_block = model_outputs_2[0]
        model_inputs_2[model_input_names_2[1]] = model_outputs_2[1]
        
        # Write to output buffer
        out_buffer[:-BLOCK_SHIFT] = out_buffer[BLOCK_SHIFT:]
        out_buffer[-BLOCK_SHIFT:] = np.zeros((BLOCK_SHIFT))
        out_buffer += np.squeeze(out_block)
        
        # Write to output file
        out_file[idx*BLOCK_SHIFT:(idx*BLOCK_SHIFT)+BLOCK_SHIFT] = out_buffer[:BLOCK_SHIFT]
    
    return out_file


def spectral_subtraction(noisy_audio, noise_profile=None, alpha=2.0):
    """Traditional spectral subtraction method"""
    # STFT parameters
    nperseg = 512
    noverlap = 384
    
    # Compute STFT
    f, t, Zxx = signal.stft(noisy_audio, fs=SAMPLE_RATE, nperseg=nperseg, noverlap=noverlap)
    
    # Estimate noise spectrum (use first 10% of signal if no profile provided)
    if noise_profile is None:
        noise_frames = int(0.1 * Zxx.shape[1])
        noise_spectrum = np.mean(np.abs(Zxx[:, :noise_frames]) ** 2, axis=1, keepdims=True)
    else:
        _, _, noise_stft = signal.stft(noise_profile, fs=SAMPLE_RATE, nperseg=nperseg, noverlap=noverlap)
        noise_spectrum = np.mean(np.abs(noise_stft) ** 2, axis=1, keepdims=True)
    
    # Spectral subtraction
    magnitude = np.abs(Zxx)
    phase = np.angle(Zxx)
    
    # Subtract noise
    clean_magnitude = np.sqrt(np.maximum(magnitude ** 2 - alpha * noise_spectrum, 0.01 * magnitude ** 2))
    
    # Reconstruct signal
    clean_stft = clean_magnitude * np.exp(1j * phase)
    _, clean_audio = signal.istft(clean_stft, fs=SAMPLE_RATE, nperseg=nperseg, noverlap=noverlap)
    
    # Match length
    if len(clean_audio) > len(noisy_audio):
        clean_audio = clean_audio[:len(noisy_audio)]
    elif len(clean_audio) < len(noisy_audio):
        clean_audio = np.pad(clean_audio, (0, len(noisy_audio) - len(clean_audio)))
    
    return clean_audio


def wiener_filter(noisy_audio, noise_profile=None):
    """Traditional Wiener filtering method"""
    # STFT parameters
    nperseg = 512
    noverlap = 384
    
    # Compute STFT
    f, t, Zxx = signal.stft(noisy_audio, fs=SAMPLE_RATE, nperseg=nperseg, noverlap=noverlap)
    
    # Estimate noise power spectrum
    if noise_profile is None:
        noise_frames = int(0.1 * Zxx.shape[1])
        noise_power = np.mean(np.abs(Zxx[:, :noise_frames]) ** 2, axis=1, keepdims=True)
    else:
        _, _, noise_stft = signal.stft(noise_profile, fs=SAMPLE_RATE, nperseg=nperseg, noverlap=noverlap)
        noise_power = np.mean(np.abs(noise_stft) ** 2, axis=1, keepdims=True)
    
    # Wiener filter
    noisy_power = np.abs(Zxx) ** 2
    wiener_gain = np.maximum(1 - noise_power / (noisy_power + 1e-10), 0.1)
    
    # Apply filter
    clean_stft = Zxx * wiener_gain
    
    # Reconstruct signal
    _, clean_audio = signal.istft(clean_stft, fs=SAMPLE_RATE, nperseg=nperseg, noverlap=noverlap)
    
    # Match length
    if len(clean_audio) > len(noisy_audio):
        clean_audio = clean_audio[:len(noisy_audio)]
    elif len(clean_audio) < len(noisy_audio):
        clean_audio = np.pad(clean_audio, (0, len(noisy_audio) - len(clean_audio)))
    
    return clean_audio


def calculate_metrics(clean, processed, fs=16000):
    """Calculate evaluation metrics: STOI, PESQ, MSE, MRE"""
    metrics = {}
    
    # Ensure same length
    min_len = min(len(clean), len(processed))
    clean = clean[:min_len]
    processed = processed[:min_len]
    
    try:
        # STOI (Short-Time Objective Intelligibility)
        metrics['stoi'] = stoi(clean, processed, fs, extended=False)
    except Exception as e:
        metrics['stoi'] = f"Error: {str(e)}"
    
    try:
        # PESQ (Perceptual Evaluation of Speech Quality)
        # PESQ requires specific sample rates: 8000 or 16000
        if fs == 16000:
            metrics['pesq'] = pesq(fs, clean, processed, 'wb')  # wideband
        elif fs == 8000:
            metrics['pesq'] = pesq(fs, clean, processed, 'nb')  # narrowband
        else:
            metrics['pesq'] = "N/A (requires 8kHz or 16kHz)"
    except Exception as e:
        metrics['pesq'] = f"Error: {str(e)}"
    
    # MSE (Mean Squared Error)
    metrics['mse'] = np.mean((clean - processed) ** 2)
    
    # MRE (Mean Relative Error)
    epsilon = 1e-10
    metrics['mre'] = np.mean(np.abs(clean - processed) / (np.abs(clean) + epsilon))
    
    return metrics


def generate_spectrogram(audio, fs, title, filename):
    """Generate and save spectrogram with gamma colormap"""
    plt.figure(figsize=(10, 4))
    plt.specgram(audio, Fs=fs, NFFT=512, noverlap=256, cmap='magma')
    plt.title(title, fontsize=12, fontweight='bold')
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (s)')
    plt.colorbar(label='Intensity (dB)')
    plt.tight_layout()
    
    filepath = os.path.join('static/spectrograms', filename)
    plt.savefig(filepath, dpi=100, bbox_inches='tight')
    plt.close()
    
    return filepath


def generate_combined_spectrogram(clean_audio, noise_audio, mixed_audio, processed_audio, fs, filename):
    """Generate combined 2x2 spectrogram"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Audio Processing Spectrograms', fontsize=16, fontweight='bold')
    
    # Clean Audio
    axes[0, 0].specgram(clean_audio, Fs=fs, NFFT=512, noverlap=256, cmap='magma')
    axes[0, 0].set_title('Clean Audio', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Frequency (Hz)')
    axes[0, 0].set_xlabel('Time (s)')
    
    # Noise
    axes[0, 1].specgram(noise_audio, Fs=fs, NFFT=512, noverlap=256, cmap='magma')
    axes[0, 1].set_title('Noise', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Frequency (Hz)')
    axes[0, 1].set_xlabel('Time (s)')
    
    # Mixed Audio
    axes[1, 0].specgram(mixed_audio, Fs=fs, NFFT=512, noverlap=256, cmap='magma')
    axes[1, 0].set_title('Mixed Audio (Noisy)', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Frequency (Hz)')
    axes[1, 0].set_xlabel('Time (s)')
    
    # Processed Audio
    axes[1, 1].specgram(processed_audio, Fs=fs, NFFT=512, noverlap=256, cmap='magma')
    axes[1, 1].set_title('Processed Audio (Denoised)', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('Frequency (Hz)')
    axes[1, 1].set_xlabel('Time (s)')
    
    plt.tight_layout()
    
    filepath = os.path.join('static/spectrograms', filename)
    plt.savefig(filepath, dpi=120, bbox_inches='tight')
    plt.close()
    
    return filepath


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/process', methods=['POST'])
def process_audio():
    try:
        # Get form data
        method = request.form.get('method', 'deterministic')
        snr_db = float(request.form.get('snr', 10))
        noise_type = request.form.get('noise_type', 'gaussian')
        traditional_method = request.form.get('traditional_method', 'spectral_subtraction')
        
        # Check files
        if 'clean_audio' not in request.files:
            return jsonify({'error': 'No clean audio file uploaded'}), 400
        
        clean_file = request.files['clean_audio']
        
        if clean_file.filename == '':
            return jsonify({'error': 'No clean audio file selected'}), 400
        
        if not allowed_file(clean_file.filename):
            return jsonify({'error': 'Invalid file format. Use WAV, FLAC, or MP3'}), 400
        
        # Generate unique ID for this processing session
        session_id = str(uuid.uuid4())
        
        # Save clean audio
        clean_filename = secure_filename(f"{session_id}_clean.wav")
        clean_path = os.path.join(app.config['UPLOAD_FOLDER'], clean_filename)
        clean_file.save(clean_path)
        
        # Load clean audio
        clean_audio, fs = load_audio(clean_path, SAMPLE_RATE)
        
        # Process based on method
        if method == 'deterministic':
            # Deterministic: Use uploaded noise file with fixed SNR
            if 'noise_audio' not in request.files:
                return jsonify({'error': 'Deterministic method requires noise audio file'}), 400
            
            noise_file = request.files['noise_audio']
            if noise_file.filename == '':
                return jsonify({'error': 'No noise audio file selected'}), 400
            
            noise_filename = secure_filename(f"{session_id}_noise.wav")
            noise_path = os.path.join(app.config['UPLOAD_FOLDER'], noise_filename)
            noise_file.save(noise_path)
            
            # Load noise
            noise_audio, _ = load_audio(noise_path, SAMPLE_RATE)
            
            # Mix with fixed SNR
            mixed_audio, used_noise = mix_audio_with_snr(clean_audio, noise_audio, snr_db)
            
            method_description = f"Deterministic (Fixed Noise, SNR={snr_db}dB)"
            
        elif method == 'stochastic':
            # Stochastic: Random noise types
            if noise_type == 'gaussian':
                mixed_audio, used_noise = add_gaussian_noise(clean_audio, snr_db)
                noise_desc = f"Gaussian Noise (SNR={snr_db}dB)"
            elif noise_type == 'white':
                mixed_audio, used_noise = add_white_noise(clean_audio, snr_db)
                noise_desc = f"White Noise (SNR={snr_db}dB)"
            elif noise_type == 'random_snr':
                # Random SNR between -5 and 20 dB
                random_snr = np.random.uniform(-5, 20)
                mixed_audio, used_noise = add_gaussian_noise(clean_audio, random_snr)
                noise_desc = f"Random SNR Gaussian ({random_snr:.1f}dB)"
            elif noise_type == 'mixed':
                # Mix of different noise types
                noise_choice = np.random.choice(['gaussian', 'white'])
                random_snr = np.random.uniform(0, 15)
                if noise_choice == 'gaussian':
                    mixed_audio, used_noise = add_gaussian_noise(clean_audio, random_snr)
                else:
                    mixed_audio, used_noise = add_white_noise(clean_audio, random_snr)
                noise_desc = f"Mixed Random ({noise_choice.title()}, {random_snr:.1f}dB)"
            else:
                mixed_audio, used_noise = add_gaussian_noise(clean_audio, snr_db)
                noise_desc = f"Gaussian Noise (SNR={snr_db}dB)"
            
            method_description = f"Stochastic ({noise_desc})"
            
        elif method == 'traditional':
            # Traditional: DSP methods
            if 'noise_audio' in request.files and request.files['noise_audio'].filename != '':
                noise_file = request.files['noise_audio']
                noise_filename = secure_filename(f"{session_id}_noise.wav")
                noise_path = os.path.join(app.config['UPLOAD_FOLDER'], noise_filename)
                noise_file.save(noise_path)
                noise_audio, _ = load_audio(noise_path, SAMPLE_RATE)
                mixed_audio, used_noise = mix_audio_with_snr(clean_audio, noise_audio, snr_db)
            else:
                # Use synthetic noise
                mixed_audio, used_noise = add_gaussian_noise(clean_audio, snr_db)
            
            # Apply traditional processing
            if traditional_method == 'spectral_subtraction':
                processed_audio = spectral_subtraction(mixed_audio, used_noise)
                trad_method_name = "Spectral Subtraction"
            elif traditional_method == 'wiener':
                processed_audio = wiener_filter(mixed_audio, used_noise)
                trad_method_name = "Wiener Filter"
            else:
                processed_audio = spectral_subtraction(mixed_audio, used_noise)
                trad_method_name = "Spectral Subtraction"
            
            method_description = f"Traditional ({trad_method_name}, SNR={snr_db}dB)"
            
            # Save outputs for traditional
            mixed_filename = f"{session_id}_mixed.wav"
            processed_filename = f"{session_id}_processed.wav"
            mixed_path = os.path.join(app.config['OUTPUT_FOLDER'], mixed_filename)
            processed_path = os.path.join(app.config['OUTPUT_FOLDER'], processed_filename)
            
            sf.write(mixed_path, mixed_audio, SAMPLE_RATE)
            sf.write(processed_path, processed_audio, SAMPLE_RATE)
            
            # Generate combined spectrogram
            generate_combined_spectrogram(clean_audio, used_noise, mixed_audio, processed_audio, 
                                        SAMPLE_RATE, f'{session_id}_combined_spec.png')
            
            # Calculate metrics
            metrics = calculate_metrics(clean_audio, processed_audio, SAMPLE_RATE)
            
            return jsonify({
                'success': True,
                'method': method_description,
                'session_id': session_id,
                'mixed_audio': f'/output/{mixed_filename}',
                'processed_audio': f'/output/{processed_filename}',
                'spectrogram_combined': f'/static/spectrograms/{session_id}_combined_spec.png',
                'metrics': metrics
            })
        
        # For deterministic and stochastic: Process with DTLN
        if method in ['deterministic', 'stochastic']:
            processed_audio = process_dtln(mixed_audio, SAMPLE_RATE)
            
            # Save outputs
            mixed_filename = f"{session_id}_mixed.wav"
            processed_filename = f"{session_id}_processed.wav"
            mixed_path = os.path.join(app.config['OUTPUT_FOLDER'], mixed_filename)
            processed_path = os.path.join(app.config['OUTPUT_FOLDER'], processed_filename)
            
            sf.write(mixed_path, mixed_audio, SAMPLE_RATE)
            sf.write(processed_path, processed_audio, SAMPLE_RATE)
            
            # Generate combined spectrogram
            generate_combined_spectrogram(clean_audio, used_noise, mixed_audio, processed_audio, 
                                        SAMPLE_RATE, f'{session_id}_combined_spec.png')
            
            # Calculate metrics
            metrics = calculate_metrics(clean_audio, processed_audio, SAMPLE_RATE)
            
            return jsonify({
                'success': True,
                'method': method_description,
                'session_id': session_id,
                'mixed_audio': f'/output/{mixed_filename}',
                'processed_audio': f'/output/{processed_filename}',
                'spectrogram_combined': f'/static/spectrograms/{session_id}_combined_spec.png',
                'metrics': metrics
            })
        
    except Exception as e:
        print(f"Error: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500


@app.route('/output/<filename>')
def output_file(filename):
    """Serve output audio files"""
    return send_file(os.path.join(app.config['OUTPUT_FOLDER'], filename))


if __name__ == '__main__':
    print("Starting DTLN Multi-Method Evaluation Web App...")
    print("Make sure model_1.onnx and model_2.onnx are in the pretrained_model/ directory")
    app.run(debug=True, host='0.0.0.0', port=5000)
