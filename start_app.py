import subprocess
import sys

# Start Streamlit server
cmd = [
    r"C:\Users\HP\miniconda3\Scripts\conda.exe",
    "run", "-n", "ml_env",
    "streamlit", "run", "app.py",
    "--server.port", "8501",
    "--server.headless", "true"
]

print("Starting Streamlit app...")
print("Command:", " ".join(cmd))
print("="*50)

try:
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    # Print initial output
    for _ in range(20):  # Read first 20 lines
        line = process.stdout.readline()
        if not line:
            break
        print(line.strip())
        if "Local URL:" in line or "Network URL:" in line:
            break
    
    print("\n" + "="*50)
    print("Streamlit app should be running!")
    print("If you see a Local URL above, the app started successfully.")
    print("You can access it in your browser.")
    
except Exception as e:
    print(f"Error starting Streamlit: {e}")