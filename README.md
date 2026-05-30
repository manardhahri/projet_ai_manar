# DermoAI

DermoAI is a Flask web application for the TP "Developpement d'une Application Web IA".
It lets a health professional log in, upload a skin lesion image, classify it with a VGG16
model, store the diagnosis history, and review patient records.

## Features

- Authentication with the default account `admin / 1234`
- Protected dashboard with analysis statistics
- `/predict` route for patient data, image upload, preprocessing, prediction, and saving
- `/patients` route for the diagnosis history
- Grad-CAM visual explanations when TensorFlow and the model are available
- SQLite auto-setup for a simple local demo
- `database.sql` included for the requested MySQL schema

## Project Structure

```text
app.py
database.sql
requirements.txt
model/
  mon_modele_melanome_vgg16.h5
static/
  style.css
  uploads/
templates/
  _messages.html
  _nav.html
  dashboard.html
  login.html
  patients.html
  predict.html
  result.html
```

## Run

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000`, then log in with `admin / 1234`.

If TensorFlow is not installed, the app still opens in demo mode so the interface can be
tested. Install the dependencies above to use the real `.h5` model.
