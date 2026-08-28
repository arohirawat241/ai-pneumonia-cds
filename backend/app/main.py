import sys
import os
import io
import base64
import shutil
import secrets
from datetime import datetime, timedelta
from typing import List, Any

# --- Path Setup ---
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy.orm import Session
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# Security Imports
from jose import JWTError, jwt
from passlib.context import CryptContext

# Import our custom modules
from backend.app.database import engine, get_db
from backend.app.models import Base, Case, User
from ml_pipeline.gradcam_generator import get_model_for_gradcam, generate_heatmap_base64

# Initialize FastAPI
app = FastAPI(title="Pneumonia Detection API")

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create uploads folder
os.makedirs("backend/uploads", exist_ok=True)

# Serve uploaded images
app.mount("/uploads", StaticFiles(directory="backend/uploads"), name="uploads")

# Create database tables
Base.metadata.create_all(bind=engine)

# Model setup
device = torch.device("cpu") 
model = None

# --- Security Settings ---
SECRET_KEY = secrets.token_urlsafe(32)
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# --- Pydantic Models ---
class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    probabilities: dict
    heatmap_base64: str
    requires_review: bool  # <-- NEW: For Uncertainty Estimation

class HealthCheck(BaseModel):
    status: str
    model_loaded: bool

class CaseResponse(BaseModel):
    id: int
    filename: str
    prediction: str
    confidence: float
    prob_normal: float
    prob_pneumonia: float
    created_at: Any 
    image_path: str
    
    class Config:
        from_attributes = True

class UserCreate(BaseModel):
    username: str
    password: str

class UserLogin(BaseModel):
    username: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

# --- Helper Functions ---
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def load_model():
    global model
    model_path = "ml_pipeline/models/best_model.pth"
    
    if not os.path.exists(model_path):
        print("⚠️ Model file not found!")
        return

    print(f"🧠 Loading model from {model_path}...")
    model = get_model_for_gradcam(device)
    if model:
        print("✅ Model loaded successfully!")

@app.on_event("startup")
async def startup_event():
    load_model()

@app.get("/")
async def root():
    return {"message": "Pneumonia Detection API is running!"}

@app.get("/health", response_model=HealthCheck)
async def health_check():
    return HealthCheck(status="healthy", model_loaded=model is not None)

# --- Auth Endpoints ---
@app.post("/api/register")
def register(user: UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.username == user.username).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    
    hashed_password = get_password_hash(user.password)
    new_user = User(username=user.username, hashed_password=hashed_password)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return {"msg": "User created successfully"}

@app.post("/api/login", response_model=Token)
def login(user: UserLogin, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.username == user.username).first()
    if not db_user or not verify_password(user.password, db_user.hashed_password):
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    
    access_token = create_access_token(data={"sub": db_user.username})
    return {"access_token": access_token, "token_type": "bearer"}

# --- Prediction Endpoint ---
@app.post("/api/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet.")
        
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        print(f"📥 Received file: {file.filename}")
        image_data = await file.read()
        print(f"📊 Image size: {len(image_data)} bytes")
        
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        print(f"🖼️ Image opened: {image.size}")
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        img_tensor = transform(image).unsqueeze(0).to(device)
        print(f"📐 Tensor shape: {img_tensor.shape}")
        
        # Save image to uploads folder
        file_location = f"backend/uploads/{file.filename}"
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(io.BytesIO(image_data), buffer)
        print(f"💾 Saved image to {file_location}")
        
        with torch.no_grad():
            print("🧠 Running inference...")
            outputs = model(img_tensor)
            print(f" Outputs: {outputs.shape}")
            
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence, predicted = torch.max(probabilities, 0)
            classes = ["NORMAL", "PNEUMONIA"]
            
            prediction_label = classes[predicted.item()]
            conf_val = round(confidence.item() * 100, 2)
            prob_norm = round(probabilities[0].item() * 100, 2)
            prob_pneu = round(probabilities[1].item() * 100, 2)
            
            print(f"✅ Prediction: {prediction_label} ({conf_val}%)")
            
            # --- UNCERTAINTY ESTIMATION ---
            requires_review = conf_val < 80.0
            # ------------------------------
            
            # Save to database
            new_case = Case(
                filename=file.filename,
                image_path=file_location,
                prediction=prediction_label,
                confidence=conf_val,
                prob_normal=prob_norm,
                prob_pneumonia=prob_pneu
            )
            db.add(new_case)
            db.commit()
            db.refresh(new_case)
            print(f"✅ Saved Case ID: {new_case.id} to database.")
            
            # Generate Grad-CAM
            print("🎨 Generating Grad-CAM heatmap...")
            heatmap_base64 = generate_heatmap_base64(image_data, model, device)
            print("✅ Heatmap generated successfully!")
            
            return PredictionResponse(
                prediction=prediction_label,
                confidence=conf_val,
                probabilities={
                    "NORMAL": prob_norm,
                    "PNEUMONIA": prob_pneu
                },
                heatmap_base64=heatmap_base64,
                requires_review=requires_review  # <-- NEW
            )
    
    except Exception as e:
        import traceback
        print("❌ ERROR DETAILS:")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

# --- History Endpoint ---
@app.get("/api/cases", response_model=List[CaseResponse])
async def get_cases(db: Session = Depends(get_db)):
    cases = db.query(Case).order_by(Case.id.desc()).limit(20).all()
    return cases

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)