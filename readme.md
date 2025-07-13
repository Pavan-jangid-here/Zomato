# 🍽️ Zomato Restaurant Intelligence App

This project is an **end-to-end machine learning web application** that predicts:

- 💰 **Estimated price** of a restaurant item  
- ⭐ **Whether the item is likely to be highly rated** by customers

The app is built using **Streamlit**, with machine learning models trained on restaurant data including features like ratings, location, cuisine, and popularity.

👉 **[Click here to experience the app](https://zomatoapps.streamlit.app/)**

---

## 📦 Project Structure

```
Zomato/
├── app.py                                      # Streamlit app UI and prediction logic
├── .gitignore                                  # Git ignore rules
├── requirements.txt                            # Python dependencies
├── data/
│   └── enhanced_zomato_dataset_clean.csv       # Original data file (used for dropdowns)
├── model/
│   ├── price_predictor.pkl                     # Trained regression model for price
│   ├── rating_classifier.pkl                   # Trained classification model for rating
│   ├── label_encoders.pkl                      # Encoders for categorical columns
│   └── Model_Creation.ipynb                    # Jupyter notebook for training the models
```

---

## ⚙️ How to Run the App

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/Zomato.git
cd Zomato
```

### 2. Create Virtual Environment (optional but recommended)

```bash
python -m venv venv
# Activate the environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit App

```bash
streamlit run app.py
```

---

## 🔍 Features

- 🔄 **Dynamic dropdowns** based on live data (`enhanced_zomato_dataset_clean.csv`)
- 🧠 **Machine learning models** trained using:
  - `RandomForestRegressor` for price prediction
  - `RandomForestClassifier` for rating classification
- 🚀 Predicts **only on button click** (no auto-refresh)
- 📊 Uses **derived features** like:
  - `Rating_Gap`
  - `Value_Score`
  - `Log_Price`

---

## 🧠 Model Training (Optional)

If you want to **retrain the models**:

1. Open `model/Model_Creation.ipynb`
2. Train the models again
3. Save the updated `.pkl` files using `joblib.dump()` in the `model/` folder

---

## 📌 To Do / Enhancements

- 💅 Visual enhancements and UI improvements

---

## 📧 Contact

Built with ❤️ by **Pavan**  
Feel free to **fork**, ⭐ **star**, or **contribute**!
