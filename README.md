# VerifyAI – Step 1: Document Upload

Real upload: frontend -> FastAPI -> saved in `uploads/`. (Dashboard/report data is still mock.)

## Requirements
Python 3.11+

## Backend (Windows)
```
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r ..\requirements.txt
uvicorn main:app --reload --port 8000
```
- Backend: http://localhost:8000
- Health check: http://localhost:8000/api/health
- Swagger test UI: http://localhost:8000/docs

## Frontend
Open the `frontend` folder in VS Code, right-click `index.html` -> **Open with Live Server**.

## Test upload
1. Go to **Verify Documents**, select up to 5 PDF/JPG/JPEG/PNG files (max 10 MB each).
2. Click **Start AI Verification**.
3. You should see "Documents uploaded successfully." and the files appear in `uploads/`.
