# 👨‍🍳 Smart Chef AI

A desktop recipe recommendation app built with Python and Tkinter. Enter the ingredients you have at home, and Smart Chef AI instantly suggests matching recipes using ML-powered TF-IDF similarity search.

---

## ✨ Features

- **Ingredient-based search** — Type ingredients you have (comma-separated) and get instant recipe matches
- **AI-powered matching** — Uses TF-IDF vectorization + cosine similarity to rank recipes by relevance
- **Recipe detail popup** — Click any result to see full ingredients and step-by-step instructions
- **Save as PDF** — Export any recipe to a beautifully formatted PDF file
- **Copy to clipboard** — Quickly copy recipe details for use anywhere
- **Clean modern UI** — Built with Tkinter using a custom green/white theme

---

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- A `recipes.csv` file (see [Dataset](#-dataset) section below)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/smart-chef-ai.git
   cd smart-chef-ai
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app**
   ```bash
   python smart_chef.py
   ```

---

## 📂 Dataset

The app expects a `recipes.csv` file in the same directory with the following columns:

| Column | Description |
|---|---|
| `Title` | Name of the recipe |
| `Cleaned_Ingredients` | Comma-separated list of ingredients |
| `Instructions` | Step-by-step cooking instructions |

A good free dataset to use: [Food.com Recipes dataset on Kaggle](https://www.kaggle.com/datasets/irkaal/foodcom-recipes-and-reviews)

---

## 📦 Dependencies

```
pandas
scikit-learn
reportlab
numpy
```

Install all at once:
```bash
pip install -r requirements.txt
```

---

## 🛠️ How It Works

1. On startup, the app loads `recipes.csv` and builds a **TF-IDF matrix** from all recipe ingredients in a background thread.
2. When you enter ingredients and click **Find Recipes**, your input is vectorized and compared against all recipes using **cosine similarity**.
3. The top 20 most relevant recipes (above a similarity threshold of 0.1) are displayed.
4. Clicking a recipe opens a detail window where you can read, copy, or export it as a PDF.

---

## 📁 Project Structure

```
smart-chef-ai/
├── smart_chef.py       # Main application file
├── requirements.txt    # Python dependencies
├── recipes.csv         # Recipe dataset (you provide this)
└── README.md
```

---

## 🙌 Contributing

Pull requests are welcome! If you find a bug or have a feature suggestion, feel free to open an issue.

---
