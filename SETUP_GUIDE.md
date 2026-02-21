# Setup Guide - AI-Powered Rental Fraud Detection System

## Quick Start (10 minutes)

### Step 1: Backend Setup

```bash
# Navigate to backend directory
cd backend

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start the backend server
python run.py
```

✅ Backend should be running at `http://localhost:8000`

### Step 2: Train ML Models (first time only)

Before using the system, train the AI models once:

```bash
# From the project root (not backend/)
cd ..
python train_all.py
```

This runs a 5-step pipeline (~2-5 minutes):

| Step | What It Does | Output |
|------|-------------|--------|
| 1 | Build BERT dataset (merge legitimate + scam texts) | `backend/data/processed/bert_dataset.csv` |
| 2 | Preprocess rental listings (feature engineering) | `backend/data/processed/*.csv` |
| 3 | Train DistilBERT fraud classifier (4 epochs) | `backend/models/bert_fraud_models/` |
| 4 | Train Isolation Forest (200 trees) | `backend/models/IsolationForest_*/` |
| 5 | Load Toronto price benchmarks | `backend/data/toronto_price_benchmarks_2026.csv` |

> **Note:** You only need to run this once. The models are saved to disk and loaded automatically when the backend starts.

### Step 3: Frontend Setup

Open a new terminal:

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start the development server
npm run dev
```

✅ Frontend should be running at `http://localhost:3000`

### Step 4: Login and Test

1. Open browser: `http://localhost:3000`
2. Login with demo credentials:

| Role | Email | Password | Portal |
|------|-------|----------|--------|
| **Admin** | `admin@rentalfraud.com` | `admin123` | `/admin` |
| **Tenant/Renter** | `renter1@example.com` | `renter123` | `/tenant` |
| **Landlord** | `landlord@example.com` | `landlord123` | `/landlord` |

## Testing the Application

### As a Tenant (Renter)

1. **Login** with tenant credentials (`renter1@example.com` / `renter123`)
2. **Browse Listings** → `/tenant/listings`
   - Click any listing card to open the full detail page
   - Use Save and Apply buttons
3. **Analyze a Listing for Fraud** → `/tenant/analyze`
   - Paste a listing description, enter price and location
   - Click "Analyze Listing" — view risk score, indicators, and explanation
4. **Verify Images** → `/tenant/verify-images`
   - Upload listing images to check for AI-generated or stolen photos
   - View forensic results with user-friendly labels and tooltips
5. **Check Address** → `/tenant/verify-address`
   - Enter an address to verify it resolves to a real residential location
6. **View History** → `/tenant/history`
   - Review past fraud analyses, export reports as HTML/PDF
7. **Manage Applications** → `/tenant/applications`
   - View your submitted applications and their statuses
8. **View Dashboard** → `/tenant`
   - See your Trust Score (engagement-weighted formula) and recent activity

### As a Landlord

1. **Login** with landlord credentials (`landlord@example.com` / `landlord123`)
2. **Create a Listing** → `/landlord/listings` → "New Listing"
   - Fill in property details (title, address, price, beds, baths, amenities)
   - Listing will appear as `pending_review` until admin approves
3. **Manage Applicants** → `/landlord/applicants`
   - View applicants for your listings
   - Click an applicant to open the **chat panel** — send/receive messages
   - Approve or reject applications
   - **When you approve**: the listing is deactivated, other applicants are auto-rejected, and a lease is auto-created with `pending_signature` status
4. **Verify Tenant Documents** → `/landlord/tenants`
   - Upload tenant documents (pay stubs, IDs, bank statements)
   - View OCR extraction results and cross-document consistency analysis
5. **Verify Property Images** → `/landlord/property-images`
   - Upload property photos for AI authenticity analysis
6. **Manage Leases** → `/landlord/leases`
   - View leases with status badges (pending_signature, active, expiring, expired)
7. **View Analytics** → `/landlord/analytics`
   - See listing performance: views, applications, Apply Rate (applications ÷ views × 100)
8. **Delete a Listing** → `/landlord/listings`
   - Delete is blocked if active or pending_signature leases exist
   - If no active leases, orphan records (messages, applications, saves, expired leases) are cleaned up automatically

### As an Administrator

1. **Login** with admin credentials (`admin@rentalfraud.com` / `admin123`)
2. **Upload Sample Dataset**:
   - Go to "Datasets" → "Upload Dataset"
   - Use the provided `sample_rental_dataset.csv` file
   - View statistics and preview

3. **Train a Model**:
   - Go to "Trained Models" → "Train New Model"
   - Select the uploaded dataset
   - Wait for training to complete (~30 seconds)
   - Review metrics (accuracy, precision, recall)

4. **Approve Listings**:
   - Go to "Listing Approval"
   - Review pending listings, approve or reject with notes

5. **Review Feedback**:
   - Go to "Feedback Review"
   - Review user fraud confirmations/denials that feed auto-learning

6. **Monitor System**:
   - Check Dashboard for statistics
   - View Audit Logs for activity tracking
   - Manage Users and their subscriptions
   - View AI Engine health status

## Features to Test

### Tenant Features
- ✅ Browse and search rental listings
- ✅ View full listing detail page (hero image, property details, amenities, price sidebar)
- ✅ Save/bookmark listings for later
- ✅ Apply to listings with a message
- ✅ Track application status (pending, viewing_scheduled, approved, rejected)
- ✅ In-app messaging with landlords (via Applicants page)
- ✅ AI fraud analysis with 4-signal fusion (BERT + Indicators + Price + Address)
- ✅ Risk score with natural language explanation and confidence
- ✅ Image verification with user-friendly forensic labels and tooltips
- ✅ Address validation via geocoding
- ✅ Analysis history with report export (HTML/PDF)
- ✅ Trust Score on dashboard and sidebar (engagement-weighted formula)
- ✅ Subscription management

### Landlord Features
- ✅ Create, edit, and delete rental listings
- ✅ View applicants with integrated chat panel (split-grid layout)
- ✅ Approve/reject applications with automatic cascades:
  - Listing deactivated on approval
  - Other pending applications auto-rejected
  - Lease auto-created with `pending_signature` status
- ✅ Delete listing protection (blocked if active leases exist)
- ✅ Lease management with status tracking (pending_signature, active, expiring, expired)
- ✅ OCR document verification (pay stubs, IDs, bank statements)
- ✅ Cross-document consistency analysis (name/address/income matching)
- ✅ Property image authenticity analysis
- ✅ Full application verification pipeline (OCR + CrossDoc + Images)
- ✅ Analytics dashboard with Apply Rate metric
- ✅ Verification history

### Admin Features
- ✅ Dataset upload and management
- ✅ Dataset preview and statistics
- ✅ Model training with metrics (Isolation Forest + BERT)
- ✅ Trained model management and versioning
- ✅ Listing approval workflow (approve/reject pending listings)
- ✅ Feedback review (user fraud confirmations feeding auto-learning)
- ✅ User management (roles, subscription plans)
- ✅ Audit logs with filtering
- ✅ AI engine health monitoring
- ✅ System analytics and monitoring dashboards

## Sample Test Cases

### High-Risk Listing Examples

```
1. Urgency Scam:
"URGENT!!! Must fill apartment TODAY! Wire deposit now or lose it! 
Owner traveling, cannot meet. Keys mailed after Western Union payment."

2. Payment Method Scam:
"Beautiful apartment $300/month. Pay with gift cards or Bitcoin only. 
Owner overseas. No viewing possible. Send payment to secure."

3. Too Good to Be True:
"Luxury 3BR penthouse downtown for only $400/month! 
Contact via WhatsApp only. Cash payment required immediately."
```

### Low-Risk Listing Examples

```
1. Legitimate Listing:
"Spacious 2BR apartment in quiet neighborhood. $1,500/month. 
Standard lease agreement. Background check required. 
Schedule viewing by calling property management office during business hours."

2. Professional Listing:
"Modern studio apartment, $1,200/month. Professionally managed. 
Online application available. Credit check and references required.
Multiple payment options accepted."
```

## Troubleshooting

### Backend Issues

**Database errors:**
```bash
# Delete and recreate database
cd backend
del rental_fraud.db          # Windows
# rm rental_fraud.db         # Mac/Linux
python run.py  # Will recreate and seed automatically
```

**Port already in use:**
- Change port in `backend/run.py` (default: 8000)

**Module not found:**
```bash
pip install -r requirements.txt
```

### Training Pipeline Issues

**`train_all.py` fails at Step 1 (Build BERT Dataset):**
- Ensure dataset files exist in `Data/selected_datasets/`
- Check that CSV files have the expected column names

**`train_all.py` fails at Step 3 (BERT Training):**
- Ensure PyTorch and Transformers are installed: `pip install torch transformers`
- BERT training works on CPU (no GPU required) but takes ~2-5 minutes
- If memory errors occur, reduce batch size in the training config

**Models not loading at runtime:**
- Verify model files exist in `backend/models/bert_fraud_models/` and `backend/models/IsolationForest_*/`
- Re-run `python train_all.py` from the project root

**Auto-learning vs retraining:**
- The auto-learning engine adjusts indicator weights and learns fraud keywords at runtime from user feedback
- It does **NOT** retrain BERT or Isolation Forest — full retraining requires re-running `train_all.py`

### Frontend Issues

**Port already in use:**
- Change port in `frontend/vite.config.js` (default: 3000)

**Dependencies error:**
```bash
# Windows:
rmdir /s /q node_modules
del package-lock.json
npm install

# Mac/Linux:
rm -rf node_modules package-lock.json
npm install
```

**API connection issues:**
- Ensure backend is running on port 8000
- Check `frontend/src/services/api.js` for correct API URL

## API Documentation

Once backend is running, visit:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## Directory Structure

```
FARUD/
├── train_all.py                # ML training pipeline (run once before deployment)
├── run_tests.py                # Test pipeline runner (backend + frontend)
├── sample_rental_dataset.csv   # Sample data for testing
├── SYSTEM_ARCHITECTURE.md      # Full architecture documentation
├── SETUP_GUIDE.md              # This file
│
├── backend/
│   ├── application/
│   │   └── use_cases/          # AI engines & business logic
│   │       ├── fraud_detection_use_cases.py    # 4-signal fusion orchestrator
│   │       ├── bert_fraud_classifier.py        # DistilBERT classifier
│   │       ├── indicator_engine.py             # Rule-based fraud patterns
│   │       ├── price_anomaly_engine.py         # Statistical price analysis
│   │       ├── address_validation_engine.py    # Geocoding validation
│   │       ├── real_image_engine.py            # Image forensics
│   │       ├── ocr_engine.py                   # OCR document analysis
│   │       ├── cross_document_engine.py        # Cross-doc consistency
│   │       ├── message_analysis_engine.py      # Message risk analysis
│   │       ├── real_xai_engine.py              # Explainable AI (IG + SHAP)
│   │       ├── explainability_engine.py        # Counterfactual analysis
│   │       ├── auto_learning_engine.py         # Runtime weight calibration
│   │       ├── data_preprocessing_pipeline.py  # 9-step data pipeline
│   │       ├── model_use_cases.py              # Isolation Forest training
│   │       └── dataset_use_cases.py            # Dataset management
│   ├── domain/
│   │   └── entities.py         # Domain models
│   ├── infrastructure/
│   │   ├── database.py         # SQLAlchemy models (14 tables)
│   │   └── seed_data.py        # Demo users & sample data
│   ├── presentation/
│   │   ├── routes/             # API endpoint groups
│   │   │   ├── auth_routes.py       # Authentication (register/login)
│   │   │   ├── admin_routes.py      # Admin operations
│   │   │   ├── renter_routes.py     # Tenant fraud analysis
│   │   │   ├── landlord_routes.py   # Document/tenant verification
│   │   │   └── property_routes.py   # Listings, applications, leases, messaging
│   │   ├── schemas.py          # Pydantic request/response models
│   │   └── dependencies.py    # JWT auth & dependency injection
│   ├── models/                 # Trained model artifacts
│   │   ├── bert_fraud_models/  # DistilBERT weights
│   │   └── IsolationForest_*/  # Isolation Forest + scaler
│   ├── data/                   # Datasets & processed files
│   │   ├── processed/          # Feature-engineered CSVs
│   │   └── uploads/            # User-uploaded files
│   ├── tests/                  # Backend test suite (pytest)
│   ├── config.py               # Configuration
│   ├── main.py                 # FastAPI app
│   ├── run.py                  # Server entry point
│   └── requirements.txt        # Python dependencies
│
├── frontend/
│   ├── src/
│   │   ├── components/         # Reusable components
│   │   │   ├── guards/         # RoleRoute, PublicRoute
│   │   │   └── layouts/        # TenantLayout, LandlordLayout, AdminLayout
│   │   ├── pages/
│   │   │   ├── public/         # LandingPage, GetStarted
│   │   │   ├── tenant/         # 12 tenant pages
│   │   │   ├── landlord/       # 13 landlord pages
│   │   │   └── admin/          # 12 admin pages
│   │   ├── services/
│   │   │   └── api.js          # Axios API client (adminAPI, renterAPI, landlordAPI)
│   │   ├── store/
│   │   │   ├── authStore.js    # Auth state (Zustand)
│   │   │   └── themeStore.js   # Theme state (Zustand)
│   │   ├── App.jsx             # Router with 41 routes
│   │   └── main.jsx            # Entry point
│   ├── package.json            # Node dependencies
│   ├── vite.config.js          # Vite config (port 3000)
│   └── tailwind.config.js      # Tailwind CSS config
│
├── Data/                       # Raw datasets
│   └── selected_datasets/      # Curated training data
│
└── test-reports/               # Test output (JUnit XML + summary)
```

## Next Steps

1. ✅ Browse listings and submit applications as a Tenant
2. ✅ Upload documents and view cross-document consistency
3. ✅ Create listings and manage applicants as a Landlord
4. ✅ Approve an applicant and verify cascades (auto-reject others, auto-create lease)
5. ✅ Delete a listing and verify cascade cleanup
6. ✅ Analyze listings with the 4-signal fraud detection pipeline
7. ✅ Review admin dashboard analytics and audit logs
8. ✅ Train models with `train_all.py` and compare metrics
9. ✅ Test messaging between Tenant and Landlord via the Applicants chat panel
10. ✅ Explore API docs at http://localhost:8000/docs

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review API documentation at http://localhost:8000/docs
3. Check browser console (F12) for frontend errors
4. Check terminal running `run.py` for backend errors
5. Review `SYSTEM_ARCHITECTURE.md` for detailed system design

## Production Deployment

For production deployment:
1. Change `SECRET_KEY` in `backend/config.py` to a strong random value
2. Use PostgreSQL instead of SQLite (`DATABASE_URL` in config)
3. Set `CORS_ORIGINS` to your actual domain(s) only
4. Enable HTTPS via a reverse proxy (Nginx / Caddy)
5. Use environment-specific configuration (`.env` files)
6. Set up logging and monitoring (e.g., Sentry, Prometheus)
7. Run `train_all.py` with production datasets before deployment
8. Disable `seed_data.py` auto-seeding in production

Enjoy testing the AI-Powered Rental Fraud Detection System! 🚀

*Last updated: February 2026*

