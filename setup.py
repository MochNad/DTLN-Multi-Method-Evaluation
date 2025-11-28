"""
Quick Setup Script untuk DTLN Multi-Method Evaluation
Menjalankan script ini untuk setup environment dan menjalankan aplikasi
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run shell command with description"""
    print(f"\n{'='*60}")
    print(f"🔧 {description}")
    print(f"{'='*60}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        print(e.stderr)
        return False

def check_models():
    """Check if ONNX models exist"""
    model_1_path = os.path.join('pretrained_model', 'model_1.onnx')
    model_2_path = os.path.join('pretrained_model', 'model_2.onnx')
    model_1 = os.path.exists(model_1_path)
    model_2 = os.path.exists(model_2_path)
    
    if not model_1 or not model_2:
        print("\n" + "="*60)
        print("⚠️  WARNING: ONNX Models Not Found!")
        print("="*60)
        print("Anda memerlukan file berikut di folder pretrained_model/:")
        if not model_1:
            print(f"  ❌ {model_1_path} - NOT FOUND")
        else:
            print(f"  ✅ {model_1_path} - FOUND")
        
        if not model_2:
            print(f"  ❌ {model_2_path} - NOT FOUND")
        else:
            print(f"  ✅ {model_2_path} - FOUND")
        
        print("\nDownload dari: https://github.com/breizhn/DTLN")
        print("Letakkan kedua file di folder pretrained_model/")
        print("="*60)
        return False
    
    print("\n✅ Model ONNX ditemukan di pretrained_model/!")
    return True

def main():
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║   DTLN Multi-Method Audio Evaluation - Setup Script     ║
    ║                                                          ║
    ║   Metode: Deterministik, Stokastik, Tradisional        ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    # Check Python version
    print(f"🐍 Python Version: {sys.version}")
    
    # Check models
    models_ok = check_models()
    
    # Install dependencies
    print("\n📦 Installing Dependencies...")
    if not run_command("pip install -r requirements.txt", "Installing Python packages"):
        print("\n⚠️  Beberapa package mungkin gagal diinstall.")
        print("Coba install manual: pip install -r requirements.txt")
    
    # Create directories
    print("\n📁 Creating directories...")
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)
    os.makedirs('static/spectrograms', exist_ok=True)
    os.makedirs('pretrained_model', exist_ok=True)
    print("✅ Directories created!")
    
    # Final check
    print("\n" + "="*60)
    print("🎯 SETUP SUMMARY")
    print("="*60)
    
    if models_ok:
        print("✅ ONNX Models: OK")
    else:
        print("❌ ONNX Models: MISSING (download required)")
    
    print("✅ Dependencies: Installed")
    print("✅ Directories: Created")
    
    print("\n" + "="*60)
    print("🚀 READY TO RUN")
    print("="*60)
    
    if models_ok:
        print("\nUntuk menjalankan aplikasi:")
        print("  python app.py")
        print("\nAtau jalankan sekarang? (y/n): ", end='')
        
        response = input().strip().lower()
        if response == 'y':
            print("\n🚀 Starting Flask application...")
            subprocess.run("python app.py", shell=True)
    else:
        print("\n⚠️  Harap download model ONNX terlebih dahulu!")
        print("Setelah model tersedia, jalankan: python app.py")

if __name__ == '__main__':
    main()
