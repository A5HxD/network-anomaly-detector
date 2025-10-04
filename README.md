# 🚨 Network Anomaly Detector (UNSW-NB15)

Hey there 👋,  
This is a project I built around the **UNSW-NB15 dataset (2015)** to detect anomalies (cyber attacks) in network traffic using Machine Learning.

---

## 📌 What this project does
- Loads the **UNSW-NB15 dataset** (train + test sets).  
- Preprocesses features (encodes categorical stuff like protocols, scales numeric values).  
- Trains a **RandomForestClassifier** to detect attack categories.  
- Evaluates model with precision, recall, F1-score + confusion matrix.  
- Lets you **predict a single traffic record** (either from the dataset or custom input).  
- Includes a **Streamlit dashboard** where you can:
  - Explore the dataset (EDA)
  - Check model evaluation (reports + plots)
  - Predict attacks by entering sample values

---

## 📥 Dataset
This project uses the **UNSW-NB15 dataset (2015)** created by UNSW Canberra Cyber.  

Download here 👉 [UNSW-NB15 Dataset](https://research.unsw.edu.au/projects/unsw-nb15-dataset)  

Once downloaded, place the CSV files like this:

```
data/UNSW_NB15_training-set.csv
data/UNSW_NB15_testing-set.csv
```

⚠️ The dataset is **large** (~250k rows), so it’s not included in this repo.

---

## ⚙️ Setup & Run

### 1. Clone repo
```bash
git clone https://github.com/YOUR_USERNAME/network-anomaly-detector.git
cd network-anomaly-detector
```

### 2. Create venv + install dependencies
```bash
python -m venv .venv
source .venv/bin/activate    # macOS/Linux
.venv\Scripts\activate     # Windows

pip install -r requirements.txt
```

### 3. Train model
```bash
python src/train.py
```
This will save the trained model into `models/anomaly_model.pkl`.

### 4. Evaluate
```bash
python src/evaluate.py
```
You’ll see a classification report + confusion matrix.

### 5. Predict sample
```bash
python src/predict.py
```

### 6. Launch Streamlit dashboard 🚀
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 📊 Features
- Multi-class attack detection (Normal, DoS, Backdoor, Reconnaissance, etc.)  
- Data exploration (attack distribution plots)  
- Model evaluation with confusion matrix  
- Streamlit dashboard with **3 pages**:
  1. 📊 Data Explorer  
  2. 🤖 Model Evaluation  
  3. 🔮 Predict Single Sample  

---

## 📝 Notes
- The dataset is **imbalanced** (some attacks are rare). A `class_weight="balanced"` option helps the model.  
- For faster training, you can train on a subset by editing `train.py` (`sample_frac=0.2`).  
- This project is mostly educational, but could be extended into a real-world IDS/IPS.

---

## 🙌 Credits
- Dataset: [UNSW-NB15](https://research.unsw.edu.au/projects/unsw-nb15-dataset)  
- ML: Scikit-learn  
- Dashboard: Streamlit  

---

💡 That’s it. If you run into issues, open an issue or just hack around 🙂  
