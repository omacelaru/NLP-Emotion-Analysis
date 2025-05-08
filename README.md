# Analiza Emoțiilor în Text

Această aplicație permite analiza emoțiilor prezente într-un text folosind modele de procesare a limbajului natural (NLP).

## Funcționalități

- Analiză emoțională a textului introdus direct
- Analiză emoțională a textului din fișiere încărcate
- Vizualizare a scorurilor emoționale
- Interfață web intuitivă și ușor de folosit

## Instalare

1. Clonează repository-ul:
```bash
git clone [URL_REPOSITORY]
cd NLP-Emotion-Analysis
```

2. Creează un mediu virtual și activează-l:
```bash
python -m venv .venv
source .venv/bin/activate  # Pentru Linux/Mac
# sau
.venv\Scripts\activate  # Pentru Windows
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

2. Deschide browser-ul la adresa indicată (de obicei http://localhost:8501)

3. Alege metoda de introducere a textului (direct sau fișier)

4. Analizează textul și vezi rezultatele

## Tehnologii utilizate

- Python
- Streamlit
- Transformers (Hugging Face)
- Plotly
- PyTorch

## Structura proiectului

```
NLP-Emotion-Analysis/
├── app/
│   ├── main.py           # Aplicația Streamlit
│   ├── model.py          # Logica modelului
│   └── utils.py          # Funcții utilitare
├── data/
│   ├── raw/             # Date neprelucrate
│   └── processed/       # Date prelucrate
├── models/              # Modele salvate
├── notebooks/           # Jupyter notebooks pentru experimente
├── requirements.txt     # Dependențe
└── README.md           # Documentație
```

## Contribuții

Contribuțiile sunt binevenite! Te rugăm să creezi un pull request pentru orice îmbunătățire.

## Licență

Acest proiect este licențiat sub MIT License. 