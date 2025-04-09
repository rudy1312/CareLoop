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
