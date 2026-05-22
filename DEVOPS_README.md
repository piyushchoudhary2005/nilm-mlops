# UK-DALE NILM — DevOps Setup Guide

Complete CI/CD pipeline for the NILM ML project using **Jenkins + SonarQube + Docker**.

---

## Architecture

```
Git Push
   │
   ▼
Jenkins Pipeline
   ├── 1. Checkout code
   ├── 2. Install Python deps (virtualenv)
   ├── 3. Lint (flake8)
   ├── 4. Unit Tests (pytest + coverage)
   ├── 5. SonarQube Scan → Quality Gate
   ├── 6. Train Models (saves artefacts)
   ├── 7. Docker Build
   └── 8. Docker Push → Docker Hub
```

---

## Prerequisites

Install these on your local machine:

| Tool | Version | Download |
|---|---|---|
| Docker Desktop | Latest | https://www.docker.com/products/docker-desktop |
| Git | Any | https://git-scm.com |
| Python | 3.10+ | https://python.org |

---

## Step 1 — Start the DevOps Stack

```bash
# Clone / navigate to your project folder
cd your-nilm-project/

# Start Jenkins + SonarQube + NILM App
docker-compose up -d

# Check all containers are running
docker-compose ps
```

Wait about 2 minutes for SonarQube to fully start.

| Service | URL | Default Credentials |
|---|---|---|
| Jenkins | http://localhost:8080 | admin / (see setup below) |
| SonarQube | http://localhost:9000 | admin / admin |
| NILM App (Gradio) | http://localhost:7860 | — |

---

## Step 2 — Jenkins First-Time Setup

1. Open http://localhost:8080
2. Get the initial admin password:
   ```bash
   docker exec jenkins cat /var/jenkins_home/secrets/initialAdminPassword
   ```
3. Install **suggested plugins** when prompted.
4. Create your admin user.

### Install additional plugins (Manage Jenkins → Plugins):
- **SonarQube Scanner** — for sonar-scanner integration
- **Docker Pipeline** — for Docker build/push steps
- **AnsiColor** — colored console output
- **HTML Publisher** — for coverage reports

---

## Step 3 — Configure Jenkins

### 3a. Add SonarQube server
> Manage Jenkins → Configure System → SonarQube Servers

- Name: `SonarQube`
- URL: `http://sonarqube:9000`  *(use container name, not localhost)*

### 3b. Add SonarQube token credential
1. Log in to SonarQube at http://localhost:9000 (admin/admin → change password)
2. My Account → Security → Generate Token → copy it
3. Jenkins → Manage Jenkins → Credentials → Global → Add Credential
   - Kind: **Secret text**
   - ID: `sonarqube-token`
   - Secret: *paste token*

### 3c. Add Docker Hub credential
> Jenkins → Manage Jenkins → Credentials → Global → Add Credential

- Kind: **Username with password**
- ID: `dockerhub-credentials`
- Username: your Docker Hub username
- Password: your Docker Hub password or access token

### 3d. Install SonarQube Scanner tool
> Manage Jenkins → Global Tool Configuration → SonarQube Scanner

- Name: `SonarScanner`
- Install automatically: ✅

---

## Step 4 — Create the Jenkins Pipeline Job

1. New Item → **Pipeline** → name it `nilm-ml-pipeline`
2. Under **Pipeline**:
   - Definition: **Pipeline script from SCM**
   - SCM: **Git**
   - Repository URL: your GitHub/GitLab repo URL
   - Branch: `*/main`
   - Script Path: `Jenkinsfile`
3. Save → **Build Now**

---

## Step 5 — Edit Jenkinsfile (your Docker Hub username)

Open `Jenkinsfile` and update line 27:

```groovy
IMAGE_NAME = "yourdockerhubuser/nilm-app"   // ← change this
```

---

## Pipeline Stages Explained

| Stage | What it does |
|---|---|
| **Checkout** | Pulls latest code from Git |
| **Environment** | Prints Python, Docker versions for debugging |
| **Install Deps** | `pip install -r requirements.txt` into a virtualenv |
| **Lint** | `flake8` checks for syntax errors and style issues |
| **Unit Tests** | `pytest` runs all tests in `tests/`, generates XML + HTML coverage |
| **SonarQube Scan** | Sends code + coverage to SonarQube for deep analysis |
| **Quality Gate** | Fails the build if SonarQube quality gate is not passed |
| **Train Models** | Runs `train_and_save.py` to train RF, XGBoost, SVR and save `.pkl` files |
| **Docker Build** | Builds image tagged `nilm-app:<build_number>` and `nilm-app:latest` |
| **Docker Push** | Pushes both tags to Docker Hub |

---

## Running Tests Locally

```bash
# Create virtualenv
python3 -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate

# Install deps
pip install -r requirements.txt

# Run tests with coverage
pytest tests/ -v --cov=. --cov-report=html

# Open coverage report
open htmlcov/index.html
```

---

## Running the App Standalone (without Jenkins)

```bash
# Build image
docker build -t nilm-app:local .

# Run container
docker run -p 7860:7860 nilm-app:local

# Open Gradio UI
open http://localhost:7860
```

---

## Useful Commands

```bash
# View Jenkins logs
docker-compose logs -f jenkins

# View SonarQube logs
docker-compose logs -f sonarqube

# Restart a single service
docker-compose restart jenkins

# Stop everything and remove volumes (full reset)
docker-compose down -v

# Shell into Jenkins container
docker exec -it jenkins bash

# Check disk usage of volumes
docker system df
```

---

## Project File Structure

```
nilm-project/
├── mini_project_multi_model.py   # Main ML script
├── train_and_save.py             # Training script (saves .pkl models)
├── requirements.txt              # Python dependencies
├── Dockerfile                    # Container definition
├── docker-compose.yml            # Full DevOps stack
├── Jenkinsfile                   # CI/CD pipeline definition
├── sonar-project.properties      # SonarQube config
├── tests/
│   └── test_model.py             # Unit tests (pytest)
├── models/                       # Saved model artefacts (gitignored)
└── outputs/                      # Plots and reports (gitignored)
```
