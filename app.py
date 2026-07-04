from flask import Flask, request, jsonify, render_template_string
from src.predict import EmotionPredictor

app = Flask(__name__)
predictor = EmotionPredictor.load('outputs')

EMOTION_EMOJI = {
    "HAPPY": "😄", "SAD": "😢", "ANGRY": "😡",
    "FEAR": "😨", "SURPRISE": "😲", "NEUTRAL": "😐",
}
EMOTION_COLOR = {
    "HAPPY": "#f9c74f", "SAD": "#4cc9f0", "ANGRY": "#f72585",
    "FEAR": "#7209b7", "SURPRISE": "#4361ee", "NEUTRAL": "#adb5bd",
}

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>🧠 Emotion Detection AI</title>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet"/>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Inter', sans-serif;
      background: #0d1117;
      color: #e6edf3;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      align-items: center;
      padding: 40px 20px;
    }
    .container { max-width: 720px; width: 100%; }
    h1 { font-size: 2rem; font-weight: 700; text-align: center; margin-bottom: 8px; }
    h1 span { background: linear-gradient(135deg, #6e40c9, #4cc9f0); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .subtitle { text-align: center; color: #8b949e; margin-bottom: 36px; font-size: 0.95rem; }
    textarea {
      width: 100%;
      padding: 16px;
      border-radius: 12px;
      border: 1px solid #30363d;
      background: #161b22;
      color: #e6edf3;
      font-size: 1rem;
      font-family: inherit;
      resize: vertical;
      min-height: 110px;
      outline: none;
      transition: border 0.2s;
    }
    textarea:focus { border-color: #6e40c9; }
    button {
      margin-top: 12px;
      width: 100%;
      padding: 14px;
      border: none;
      border-radius: 12px;
      background: linear-gradient(135deg, #6e40c9, #4cc9f0);
      color: white;
      font-size: 1rem;
      font-weight: 600;
      cursor: pointer;
      transition: opacity 0.2s, transform 0.1s;
    }
    button:hover { opacity: 0.9; transform: translateY(-1px); }
    button:active { transform: translateY(0); }
    .examples { margin-top: 14px; display: flex; flex-wrap: wrap; gap: 8px; }
    .example-chip {
      padding: 6px 14px;
      border-radius: 999px;
      border: 1px solid #30363d;
      background: #161b22;
      color: #8b949e;
      font-size: 0.8rem;
      cursor: pointer;
      transition: all 0.2s;
    }
    .example-chip:hover { border-color: #6e40c9; color: #e6edf3; }
    .result-box {
      margin-top: 32px;
      background: #161b22;
      border: 1px solid #30363d;
      border-radius: 16px;
      padding: 24px;
      display: none;
    }
    .top-emotion {
      display: flex;
      align-items: center;
      gap: 14px;
      margin-bottom: 24px;
    }
    .emoji { font-size: 3rem; }
    .top-label { font-size: 1.6rem; font-weight: 700; }
    .top-conf { font-size: 1rem; color: #8b949e; margin-top: 2px; }
    .bar-row { margin-bottom: 12px; }
    .bar-label {
      display: flex;
      justify-content: space-between;
      margin-bottom: 4px;
      font-size: 0.85rem;
      color: #8b949e;
    }
    .bar-label span:first-child { font-weight: 600; color: #e6edf3; }
    .bar-track {
      background: #0d1117;
      border-radius: 999px;
      height: 10px;
      overflow: hidden;
    }
    .bar-fill {
      height: 100%;
      border-radius: 999px;
      transition: width 0.6s cubic-bezier(.4,0,.2,1);
    }
    .loading { text-align: center; color: #8b949e; font-size: 0.9rem; margin-top: 16px; display: none; }
  </style>
</head>
<body>
  <div class="container">
    <h1>🧠 <span>Emotion Detection AI</span></h1>
    <p class="subtitle">Type any sentence and the NLP model will predict its underlying emotion.</p>

    <textarea id="inputText" placeholder="e.g. I'm absolutely thrilled about my new job offer!"></textarea>
    <button onclick="analyze()">⚡ Analyze Emotion</button>

    <div class="examples">
      <span class="example-chip" onclick="fill(this)">I'm so happy today!</span>
      <span class="example-chip" onclick="fill(this)">I miss her so much it hurts...</span>
      <span class="example-chip" onclick="fill(this)">This is making me furious!</span>
      <span class="example-chip" onclick="fill(this)">I can't believe they threw a surprise party!</span>
      <span class="example-chip" onclick="fill(this)">Please send the report by Friday.</span>
    </div>

    <div class="loading" id="loading">Analyzing...</div>

    <div class="result-box" id="result">
      <div class="top-emotion">
        <div class="emoji" id="topEmoji"></div>
        <div>
          <div class="top-label" id="topLabel"></div>
          <div class="top-conf" id="topConf"></div>
        </div>
      </div>
      <div id="bars"></div>
    </div>
  </div>

  <script>
    const COLORS = {{ colors|tojson }};
    const EMOJIS = {{ emojis|tojson }};

    function fill(el) {
      document.getElementById('inputText').value = el.textContent;
    }

    async function analyze() {
      const text = document.getElementById('inputText').value.trim();
      if (!text) return;
      document.getElementById('result').style.display = 'none';
      document.getElementById('loading').style.display = 'block';
      const res = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text })
      });
      const data = await res.json();
      document.getElementById('loading').style.display = 'none';
      document.getElementById('topEmoji').textContent = EMOJIS[data.emotion] || '🤔';
      document.getElementById('topLabel').textContent = data.emotion;
      document.getElementById('topLabel').style.color = COLORS[data.emotion] || '#e6edf3';
      document.getElementById('topConf').textContent = `Confidence: ${(data.confidence * 100).toFixed(1)}%`;

      const bars = document.getElementById('bars');
      bars.innerHTML = '';
      const sorted = Object.entries(data.probabilities).sort((a,b) => b[1]-a[1]);
      sorted.forEach(([label, prob]) => {
        const pct = (prob * 100).toFixed(1);
        const color = COLORS[label] || '#6e40c9';
        bars.innerHTML += `
          <div class="bar-row">
            <div class="bar-label">
              <span>${EMOJIS[label] || ''} ${label}</span>
              <span>${pct}%</span>
            </div>
            <div class="bar-track">
              <div class="bar-fill" style="width:${pct}%; background:${color};"></div>
            </div>
          </div>`;
      });
      document.getElementById('result').style.display = 'block';
    }

    document.getElementById('inputText').addEventListener('keydown', e => {
      if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); analyze(); }
    });
  </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML, colors=EMOTION_COLOR, emojis=EMOTION_EMOJI)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text', '').strip()
    if not text:
        return jsonify({'emotion': 'NEUTRAL', 'confidence': 1.0, 'probabilities': {}})
    result = predictor.predict(text)
    emotion = result.get('label', 'NEUTRAL')
    confidence = result.get('confidence', 0.0)
    probabilities = result.get('probabilities', {})
    return jsonify({
        'emotion': emotion,
        'confidence': float(confidence),
        'probabilities': {k: float(v) for k, v in probabilities.items()}
    })

if __name__ == '__main__':
    print("\n[OK] App running on port 7860 (Hugging Face Spaces ready)\n")
    app.run(host='0.0.0.0', port=7860)
