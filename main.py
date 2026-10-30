import subprocess
import sys

def main():
    print("Launching Scientific Calculator Streamlit UI...")
    subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])

if __name__ == "__main__":
    main()

