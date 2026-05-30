from flask import Flask, request, jsonify, render_template_string
import re
import string
import traceback
import joblib
import numpy as np

MODEL_PATH = "fake_news_model.pkl"
VECT_PATH = "tfidf_vectorizer.pkl"

app = Flask(__name__)

# ---------------- Text Preprocessing ----------------
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = text.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------- Model Loading ----------------
model, vectorizer = None, None
model_loaded, load_error = False, None
try:
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECT_PATH)
    model_loaded = True
    print("✅ Model loaded successfully!")
except Exception as e:
    load_error = str(e)
    print("❌ Error loading model:", load_error)


# ---------------- HTML Template (UX Optimized) ----------------
HOME_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AI Content Verification</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
* {margin: 0; padding: 0; box-sizing: border-box;}
body {
    font-family: 'Segoe UI', Roboto, sans-serif;
    background: radial-gradient(circle at 25% 10%, #0b1120, #020617 80%);
    color: #e2e8f0;
    display: flex;
    justify-content: center;
    align-items: center;
    min-height: 100vh;
    padding: 40px;
}

.container {
    width: 100%;
    max-width: 1400px; /* ⬆️ Wider Container */
    background: #1e293b;
    border-radius: 22px;
    box-shadow: 0 0 40px rgba(59,130,246,0.3);
    overflow: hidden;
    padding-bottom: 30px;
}

.header {
    text-align: center;
    padding: 40px 20px 25px;
    background: linear-gradient(90deg, #1e3a8a, #2563eb);
}

.header h1 {
    font-size: 2.8em;
    font-weight: 800;
    margin: 0;
    background: linear-gradient(270deg, #60a5fa, #34d399, #f472b6, #a78bfa);
    background-size: 800% 800%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: colorCycle 6s ease infinite;
}
@keyframes colorCycle {
    0%{background-position:0% 50%}50%{background-position:100% 50%}100%{background-position:0% 50%}
}
.header p {color: #cbd5e1; font-size: 1.05em; margin-top: 10px;}
.content {padding: 35px 50px;}

textarea {
    width: 100%;
    min-height: 500px; /* ⬆️ Much Bigger Text Box */
    background: #0f172a;
    color: #f8fafc;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 25px;
    font-size: 1.1em; /* ⬆️ Increased Font Size */
    resize: vertical;
    transition: all 0.25s;
}
textarea:focus {
    outline: none;
    border-color: #3b82f6;
    box-shadow: 0 0 15px rgba(59,130,246,0.5);
}

.btn-container {
    display: flex;
    gap: 15px;
    margin-top: 25px;
}
.btn {
    flex: 1;
    padding: 14px;
    border-radius: 10px;
    border: none;
    font-size: 1em;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.3s ease;
}
.btn-primary {background: linear-gradient(90deg, #2563eb, #1e3a8a); color: white;}
.btn-primary:hover {transform: scale(1.04); box-shadow: 0 0 20px rgba(59,130,246,0.6);}
.btn-secondary {background: #334155; color: #f1f5f9;}
.btn-secondary:hover {background: #475569;}

.loading {display: none; text-align: center; margin-top: 25px; color: #cbd5e1;}
.spinner {
    border: 4px solid #1e293b; border-top: 4px solid #3b82f6; border-radius: 50%;
    width: 40px; height: 40px; animation: spin 1s linear infinite; margin: 0 auto 12px;
}
@keyframes spin {from{transform:rotate(0)}to{transform:rotate(360deg)}}
.error {display: none; background: #7f1d1d; color: #fca5a5; padding: 14px; margin-top: 20px; border-radius: 8px;}

.result {
    display: none; margin-top: 40px; padding: 25px; /* ⬆️ Slightly increased padding for better look */
    background: #0f172a; border-radius: 16px; box-shadow: 0 0 25px rgba(37,99,235,0.25);
}
.result-container {
    display: flex; justify-content: space-around; align-items: center; gap: 30px;
}
.confidence {
    width: 90%; height: 25px; background: #1e293b; margin: 20px auto; border-radius: 9999px; overflow: hidden;
}
.fill {
    height: 100%; width: 0%; background: linear-gradient(90deg, #06b6d4, #3b82f6);
    display: flex; align-items: center; justify-content: center; color: white; font-weight: 600;
    transition: width 1.2s ease-in-out; font-size: 1em;
}
canvas {max-width: 350px; margin: 10px auto; display: block;}
</style>
</head>
<body>
<div class="container">
    <div class="header">
        <h1>🔍 Fake News Detector</h1>
        <p>Detect Real or Fake News Using Machine Learning</p>
    </div>
    <div class="content">
        {% if not model_loaded %}
        <div class="error" style="display:block;">⚠️ Model failed to load: {{ load_error }}</div>
        {% endif %}
        <form id="form">
            <textarea id="text" placeholder="Paste your article or news content here..."></textarea>
            <div class="btn-container">
                <button type="submit" class="btn btn-primary" {% if not model_loaded %}disabled{% endif %}>Analyze</button>
                <button type="button" id="clearBtn" class="btn btn-secondary">Clear</button>
            </div>
        </form>
        <div id="loading" class="loading"><div class="spinner"></div><p>AI is analyzing...</p></div>
        <div id="errorBox" class="error"></div>
        <div id="resultBox" class="result">
            <div class="result-container">
                <div>
                    <h2 id="resultTitle"></h2>
                    <div class="confidence"><div id="fill" class="fill"></div></div>
                </div>
                <canvas id="pieChart"></canvas>
            </div>
        </div>
    </div>
</div>

<script>
let chartInstance=null;
const textEl=document.getElementById("text");
const fillEl=document.getElementById("fill");
const resultBox=document.getElementById("resultBox");
const errorBox=document.getElementById("errorBox");
const loadingEl=document.getElementById("loading");

document.getElementById("form").addEventListener("submit",async(e)=>{
  e.preventDefault();
  const text=textEl.value.trim();
  if(!text)return showError("Please enter text to analyze.");
  loadingEl.style.display="block";errorBox.style.display="none";resultBox.style.display="none";
  if(chartInstance){chartInstance.destroy();}
  try{
    const res=await fetch("/predict",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({text})});
    const data=await res.json();if(!res.ok)throw new Error(data.error);displayResult(data);
  }catch(err){showError(err.message);}finally{loadingEl.style.display="none";}
});

document.getElementById("clearBtn").addEventListener("click",()=>{
  textEl.value="";resultBox.style.display="none";errorBox.style.display="none";
  if(chartInstance){chartInstance.destroy();}
  fillEl.style.width="0%";fillEl.textContent="";
});

function showError(msg){errorBox.textContent=msg;errorBox.style.display="block";}

function displayResult(data){
  const prob=(data.probability||0.5)*100;
  const isFake=data.prediction==="fake";
  const conf=prob.toFixed(1);
  fillEl.style.width=conf+"%";fillEl.textContent=conf+"%";
  document.getElementById("resultTitle").textContent=isFake?"🚨 Fake News Detected":"✅ Real News Detected";

  // Calculate the percentage of the predicted class (Real or Fake)
  const realPerc=isFake?100-prob:prob;
  const fakePerc=100-realPerc;

  const ctx=document.getElementById("pieChart").getContext("2d");
  if(chartInstance)chartInstance.destroy();

  const centerText={
    id:"centerText",
    afterDatasetsDraw(chart){
      const {ctx,chartArea:{width,height,top,left}}=chart;
      const x=left+width/2, y=top+height/2;
      ctx.save();
      ctx.font="bold 1.1em Segoe UI";
      ctx.fillStyle=isFake?"#ef4444":"#10b981";
      ctx.textAlign="center";ctx.textBaseline="middle";
      ctx.shadowBlur=10;ctx.shadowColor=ctx.fillStyle;
      
      // Use the correct percentage (realPerc or fakePerc) for display
      ctx.fillText(`${(isFake?fakePerc:realPerc).toFixed(1)}% ${(isFake?"Fake":"Real")}`,x,y);
      ctx.restore();
    }
  };

  chartInstance=new Chart(ctx,{
    type:"doughnut",
    data:{
      labels:["Real","Fake"],
      datasets:[{data:[realPerc,fakePerc],backgroundColor:["#10b981","#ef4444"],borderWidth:0,hoverOffset:8}]
    },
    options:{
      cutout:"72%",
      animation:{animateRotate:true,animateScale:true,duration:1600,easing:"easeInOutQuart"},
      plugins:{legend:{labels:{color:"#e2e8f0",font:{size:12}}},tooltip:{enabled:false}}
    },
    plugins:[centerText]
  });

  resultBox.style.display="block";
}
</script>
</body>
</html>
"""

# ---------------- Flask Routes ----------------
@app.route("/")
def home():
    return render_template_string(HOME_HTML, model_loaded=model_loaded, load_error=load_error)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if not model_loaded or model is None or vectorizer is None:
            return jsonify({"error":"Model not loaded"}),500
        data=request.get_json()
        text=clean_text(data.get("text",""))
        if not text:
            return jsonify({"error":"Empty text"}),400
        X=vectorizer.transform([text])
        pred_raw=model.predict(X)[0]
        prob=0.5
        if hasattr(model,"predict_proba"):
            probs=model.predict_proba(X)[0]
            try: idx=list(model.classes_).index(pred_raw)
            except: idx=int(np.argmax(probs))
            prob=float(probs[idx])
        prediction="real" if pred_raw in [1,"real","Real"] else "fake"
        return jsonify({"prediction":prediction,"probability":round(prob,4)})
    except Exception as e:
        print("❌",traceback.format_exc())
        return jsonify({"error":str(e)}),500

if __name__=="__main__":
    app.run(debug=True)