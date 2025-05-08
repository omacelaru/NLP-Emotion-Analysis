# Analiza Emoțiilor în Text folosind NLP

Această aplicație oferă o analiză complexă a emoțiilor din text, combinând puterea analizei de sentiment VADER cu detectarea detaliată a emoțiilor prin DistilRoBERTa. Proiectul este conceput pentru a oferi o înțelegere profundă a conținutului emoțional din text, fiind util atât pentru analiza feedback-ului utilizatorilor, cât și pentru cercetare în domeniul procesării limbajului natural.

## Caracteristici

- **Analiză Duală**: Combină VADER pentru analiza de sentiment și DistilRoBERTa pentru detectarea detaliată a emoțiilor
- **Analiză în Timp Real**: Oferă analiză instantanee pentru orice text introdus
- **Analiză Vizuală**: 
  - Indicator de sentiment pentru sentimentul general
  - Roata emoțiilor pentru distribuția detaliată
  - Analiză comparativă între modele
  - Detalii despre scoruri
- **Interfață Prietenoasă**: Interfață Streamlit curată și intuitivă

## Tehnologii Folosite

- **Frontend**: Streamlit
- **Modele NLP**: 
  - VADER (Valence Aware Dictionary and sEntiment Reasoner)
  - DistilRoBERTa (emotion-english-distilroberta-base)
- **Vizualizare**: Plotly
- **Dependențe**:
  - streamlit==1.32.0
  - transformers==4.38.2
  - torch==2.2.1
  - pandas==2.2.1
  - numpy==1.26.4
  - scikit-learn==1.4.1
  - plotly==5.19.0
  - nltk==3.8.1
  - vaderSentiment==3.3.2

## Instalare

1. Clonează repository-ul:
```bash
git clone https://github.com/yourusername/NLP-Emotion-Analysis.git
cd NLP-Emotion-Analysis
```

2. Creează și activează un mediu virtual:
```bash
python -m venv .venv
source .venv/bin/activate  # Pentru Windows: .venv\Scripts\activate
```

3. Instalează dependențele:
```bash
pip install -r requirements.txt
```

## Utilizare

1. Pornește aplicația:
```bash
streamlit run app/main.py
```

2. Deschide browser-ul la adresa locală furnizată (de obicei http://localhost:8501)

3. Introdu textul în zona de input și apasă "Analyze Text" pentru a obține:
   - Emoția dominantă
   - Intensitatea emoției
   - Detalii despre emoții
   - Reprezentări vizuale ale analizei

## Structura Proiectului

```
NLP-Emotion-Analysis/
├── app/
│   ├── core/
│   │   └── analyzer.py      # Logica principală de analiză
│   ├── utils/
│   │   ├── config.py        # Setări de configurare
│   │   └── visualizations.py # Utilități de vizualizare
│   └── main.py             # Fișierul principal al aplicației
├── requirements.txt        # Dependențe proiect
└── README.md              # Documentație proiect
```

## Cum Funcționează

1. **Introducere Text**: Utilizatorii introduc textul prin interfața Streamlit
2. **Proces de Analiză**:
   - VADER analizează sentimentul (pozitiv, negativ, neutru)
   - DistilRoBERTa detectează emoțiile specifice
3. **Afișare Rezultate**:
   - Metricile arată emoția dominantă și intensitatea
   - Vizualizările oferă o înțelegere intuitivă
   - Analiza comparativă arată diferențele dintre modele

## Contribuitori

## Resurse

- VADER Sentiment Analysis: [Repository GitHub](https://github.com/cjhutto/vaderSentiment)
- Modelul DistilRoBERTa: [Hugging Face](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base)
- Streamlit: [Documentație](https://docs.streamlit.io/) 