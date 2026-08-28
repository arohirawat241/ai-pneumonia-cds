# AI-Powered Pneumonia Detection - Clinical Decision Support System

A full-stack AI application that assists in detecting pneumonia from chest X-rays using Deep Learning and Explainable AI (Grad-CAM).

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)
![React](https://img.shields.io/badge/React-18-blue.svg)


## Features

### Core Functionality
- **Deep Learning Model**: ResNet50 trained on 5,000+ chest X-rays with 93.75% validation accuracy.
- **Explainable AI (Grad-CAM)**: Visual heatmaps showing exactly where the AI is looking in the lungs.
- **Uncertainty Estimation**: Flags low-confidence predictions (<80%) for human review - critical for medical safety.
- **Secure Authentication**: JWT-based login system with bcrypt password hashing.
- **Case Management**: SQLite database saves every analysis with patient history tracking.
- **Modern UI**: Drag-and-drop interface built with React & Material UI.

### Technical Highlights
- **Backend**: FastAPI (Python) with async support.
- **Frontend**: React 18 + TypeScript + Material UI.
- **ML Pipeline**: PyTorch, Grad-CAM, TorchVision.
- **Database**: SQLAlchemy ORM with SQLite.
- **Deployment**: Docker-ready architecture.

## Quick Start

### Prerequisites
- Python 3.10+
- Node.js 18+
- Docker (optional, for containerized deployment)

### Local Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/ai-pneumonia-cds.git
cd ai-pneumonia-cds
```

2. **Install Backend Dependencies**
```bash
cd backend
python -m venv venv
source venv/Scripts/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install bcrypt==3.2.2  # Compatibility fix
```

3. **Install Frontend Dependencies**
```bash
cd ../frontend
npm install
```
4. **Download the Dataset**
The dataset is not included in this repository due to size. Download it from:
- **Source**: [Chest X-Ray Pneumonia Dataset (Kaggle)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)
- Extract the contents into `ml_pipeline/data/raw/`

5. **Train the model**

7. **Run the Application**
```bash
# Terminal 1 - Backend
cd backend
python app/main.py

# Terminal 2 - Frontend
cd frontend
npm start
```

6. **Access the App**
- Frontend: http://localhost:3000
- Backend API Docs: http://localhost:8000/docs

### Docker Deployment

```bash
docker-compose up --build
```

## Model Performance

| Metric | Score |
|--------|-------|
| **Validation Accuracy** | 93.75% |
| **Training Accuracy** | 99.19% |
| **Architecture** | ResNet50 |
| **Dataset** | Chest X-Ray (Pneumonia) - Kaggle |
| **Framework** | PyTorch |

## Architecture

```text
+-------------+     +--------------+     +-------------+
|   React UI  |---->|  FastAPI     |---->|  ResNet50   |
| (Port 3000) |     |  (Port 8000) |     |   Model     |
+-------------+     +------+-------+     +-------------+
                           |
                           v
                    +--------------+
                    |   SQLite     |
                    |   Database   |
                    +--------------+
```

## API Endpoints

- **POST** `/api/register` - Create new user account
- **POST** `/api/login` - User authentication (returns JWT token)
- **POST** `/api/predict` - Upload X-ray and get AI prediction
- **GET** `/api/cases` - Retrieve patient case history
- **GET** `/health` - Health check endpoint

## Key Features Explained

### Grad-CAM Heatmaps
The application uses Gradient-weighted Class Activation Mapping (Grad-CAM) to generate visual explanations of the model's predictions. This helps radiologists understand why the AI made a particular diagnosis.

### Uncertainty Estimation
For medical AI, knowing when NOT to trust the model is crucial. Any prediction with confidence below 80% triggers a "Requires Human Review" warning, ensuring patient safety.

### Secure Authentication
Passwords are hashed using bcrypt before storage. JWT tokens provide stateless authentication with 30-minute expiration.

## Project Structure

```text
ai-pneumonia-cds/
├── backend/
│   ├── app/
│   │   ├── main.py          # FastAPI application
│   │   ├── models.py        # SQLAlchemy database models
│   │   └── database.py      # Database configuration
│   ├── uploads/             # Stored X-ray images
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   └── App.tsx          # Main React component
│   ── public/
├── ml_pipeline/
│   ├── models/
│   │   └── best_model.pth   # Pre-trained ResNet50
│   ├── gradcam_generator.py # Explainable AI module
│   └── data_loader.py       # PyTorch data pipeline
├── docker-compose.yml
└── README.md
```

## Future Enhancements

- [ ] PostgreSQL for production database
- [ ] Admin dashboard with analytics
- [ ] Multi-model ensemble for higher accuracy
- [ ] Integration with PACS systems (DICOM support)
- [ ] Cloud deployment (AWS/Azure)

## License

This project is for educational and research purposes only. Not for clinical use.

## Author

**Arohi Rawat**

## Acknowledgments

- Dataset: [Chest X-Ray (Pneumonia)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) by Paul Mooney
- Grad-CAM Implementation: [pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam)
```
