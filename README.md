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
## Demo
<img width="1600" height="778" alt="66bcc993-91f0-40fe-a4c7-16581e0413e5" src="https://github.com/user-attachments/assets/a0579d52-a839-41e4-952f-653cb9528722" />
<img width="1600" height="779" alt="68d7402e-5580-4e9e-94da-22207539a6f1" src="https://github.com/user-attachments/assets/eb8ec40b-82ca-494b-96af-a40e6c62b5d0" />
<img width="1600" height="776" alt="ad9a9212-50d3-4281-9d4a-0dc35023d0fd" src="https://github.com/user-attachments/assets/c3303d60-5c54-4b3c-9178-e48bb62d1b2e" />
<img width="1600" height="781" alt="3f3f5611-d126-424b-8bf4-1d12c613c0b3" src="https://github.com/user-attachments/assets/25eb4fa9-2303-4128-b04a-3640d29dcd4a" />
<img width="1600" height="774" alt="976a258a-872c-4ac6-b8d0-365365abcd27" src="https://github.com/user-attachments/assets/f5b09198-495a-41a2-885e-8df9dc5134ae" />
<img width="1600" height="777" alt="4a8c4ca8-9baf-4d93-a9e7-7037bdf1fae8" src="https://github.com/user-attachments/assets/66f858bf-29a8-402c-8f01-71a9cb916395" />

Open `http://127.0.0.1:5000`, then log in with `admin / 1234`.

If TensorFlow is not installed, the app still opens in demo mode so the interface can be
tested. Install the dependencies above to use the real `.h5` model.
