<h2>CareLoop: Post-Discharge Feedback Loop</h2>
<b>CareLoop</b> is a lightweight platform for hospitals to collect real-time feedback from discharged patients in the form of text, audio, or video. The system uses LLaMA-based ML models to auto-summarize concerns, detect sentiment, and surface weekly trends for hospital administrators.
<br/>

## Features

### For Patients
- Anonymous feedback via **text, audio, or video**
- Login Using PatientID
- Simple, clean UI inspired by Human Interface Guidelines

### For Admins
- Dashboard with:
  - Summarized patient concerns
  - Auto-generated tags (e.g., *billing*, *cleanliness*, *staff*)
  - Sentiment analysis
  - Weekly trend insights (e.g., charts showing rising issues)
- Exportable reports for internal action

---

## 📁 Folder Structure

```
CareLoop/
├── client/       # React frontend
├── server/       # Express backend
├── ml/           # ML pipeline (Whisper, transformers, analysis)
│   ├── data/     # Sample audio files, add your own files to test them.
│   ├── transcribe.ipynb
│   └── environment.yml
└── README.md
```


## 🚀 Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/spandan-mozumder/CareLoop.git
cd CareLoop
```

### 2. Install dependencies: Frontend
```bash
cd client
npm install
npm run dev
```

### 3. Install dependencies: Backend
make sure to have your own MongDB key in .env
```bash
cd server
npm install
npm run dev
```

### 4. ML/NLP Setup (requires Python 3.9 and Conda)
```bash
cd ml
conda env create -f environment.yml
conda activate careloop-ml
```

The ML file is a Jupyter Notebook, make sure your IDE supports .ipynb files or you have Jupyter notebook installed.
OpenAI Whisper doesn't support metal performance shaders on MacOS yet, using cpu instead. For liniux and windows, cuda will be used if you have an NVDIA gpu.
#### Setting up ffmpeg (required for OpenAI Whisper)
##### For windows:
<ul>
<li>Go to: https://ffmpeg.org/download.html and Download the "Essentials" ZIP file
<li>Extract it, and copy the path to the bin folder (e.g., C:\ffmpeg\bin)
<li>Add to <b>System Environment Variables > PATH</b>:
<li>Restart your Jupyter kernel or terminal and run again.
<li>To Test
  
```bash
ffmpeg -version
```
</ul>

##### For macOS (with Homebrew):

```bash
brew install ffmpeg
```

#### For Linux(Debian based):

```bash
sudo apt update && sudo apt install ffmpeg
```
You can directly view the ML outputs cell by cell in the notebook

## What's left to do?
- Integrate ML with backend
- Report Generation
- Multi Modal sentiment analysis using Voice, Text and Video
- View Transcript, tags and sentiment in Fronetned before sumbitting feedback.
