# Age & Gender Estimation - Ready for Render (no Docker)

## What this repo contains
- `app.py` : Flask app that accepts an image upload and returns age & gender estimates using DeepFace.
- `templates/index.html` : Simple upload page.
- `requirements.txt` : Python dependencies (uses `tensorflow-cpu` for Render).
- `Procfile` : Start command for Render / Heroku-style hosts.

## Quick steps to deploy to Render (no Docker)

1. Push this repo to GitHub or GitLab.
2. Go to https://render.com → New → Web Service → Connect your repo.
3. Environment: **Python** (not Docker).
4. Build Command: `pip install -r requirements.txt`
5. Start Command: `gunicorn app:app --bind 0.0.0.0:$PORT --workers 2`
6. Choose an instance with >=1GB RAM (TensorFlow needs memory). Click Create.
7. Wait for build & deploy. Open the provided `https://...onrender.com` URL and upload an image.

## Notes & troubleshooting
- **First request may be slow**: DeepFace downloads model files on first run; this can take 10-60 seconds.
- **Ephemeral storage**: Uploaded files are deleted after processing. Persistent storage requires S3 or similar.
- **If deployment fails**: Check Render build logs. Common fixes: increase instance RAM, set fewer Gunicorn workers (e.g., 1), or use a lighter model.

## To test locally
1. Create virtualenv:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   python app.py
   ```
2. Visit http://localhost:5000 and upload an image.

## If you want a lighter alternative
DeepFace + TensorFlow is convenient but heavy. Tell me if you want a version that uses a lighter model or runs inference client-side (no server ML).
