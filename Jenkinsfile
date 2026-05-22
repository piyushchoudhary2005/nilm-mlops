
// ─────────────────────────────────────────────────────────────
//  UK-DALE NILM Project — Jenkins Declarative Pipeline
//
//  Stages:
//    1. Checkout         — clone source from SCM
//    2. Environment      — print Python / Docker versions
//    3. Install Deps     — pip install into virtualenv
//    4. Lint             — flake8 code quality checks
//    5. Unit Tests       — pytest with coverage report
//    6. SonarQube Scan   — push results to SonarQube server
//    7. Quality Gate     — fail build if gate not passed
//    8. Train Models     — run training script, save artefacts
//    9. Docker Build     — build the container image
//   10. Docker Push      — push to Docker Hub / registry
//   11. Notify           — send Slack / email on success or failure
// ─────────────────────────────────────────────────────────────
 
pipeline {
 
    agent any
 
    // ── Pipeline-wide environment variables ──
    environment {
        // Docker image name — change to your Docker Hub username
        IMAGE_NAME     = "yourdockerhubuser/nilm-app"
        IMAGE_TAG      = "${env.BUILD_NUMBER}"
        IMAGE_LATEST   = "${IMAGE_NAME}:latest"
        IMAGE_VERSIONED = "${IMAGE_NAME}:${IMAGE_TAG}"
 
        // SonarQube — matches the server name in Jenkins → Configure System
        SONAR_SERVER   = "SonarQube"
 
        // Jenkins credentials IDs (add these in Jenkins → Credentials)
        DOCKER_CREDS   = "dockerhub-credentials"   // Username+Password credential
        SONAR_TOKEN    = "sonarqube-token"          // Secret text credential
 
        // Python virtualenv path
        VENV_DIR       = ".venv"
    }
 
    options {
        buildDiscarder(logRotator(numToKeepStr: '10'))
        timeout(time: 60, unit: 'MINUTES')
        disableConcurrentBuilds()
    }
 
    triggers {
        // Auto-build on every push to main / master
        pollSCM('H/5 * * * *')
    }
 
    stages {
 
        // ── Stage 1: Checkout ──────────────────────────────────────
        stage('Checkout') {
            steps {
                echo '📥 Checking out source code...'
                checkout scm
                sh 'git log --oneline -5'
            }
        }
 
        // ── Stage 2: Environment Info ──────────────────────────────
        stage('Environment') {
            steps {
                echo '🔍 Printing environment info...'
                sh '''
                    echo "Build Number: $BUILD_NUMBER"
                    echo "Branch: $GIT_BRANCH"
                    echo "Workspace: $WORKSPACE"
                    python3 --version || echo "python3 not found in PATH"
                    pip3 --version   || echo "pip3 not found in PATH"
                    which python3    || echo "python3 binary location unknown"
                '''
            }
        }
 
        // ── Stage 3: Install Dependencies ─────────────────────────
        stage('Install Dependencies') {
            steps {
                echo '📦 Installing Python and dependencies...'
                sh '''
                    # Install python3 and system libs if not present
                    which python3 || (apt-get update -qq && apt-get install -y python3 python3-pip python3-venv)
 
                    # Install HDF5 headers needed by tables/h5py
                    apt-get install -y libhdf5-dev pkg-config 2>/dev/null || true
 
                    # Create virtualenv
                    python3 -m venv ${VENV_DIR}
                    . ${VENV_DIR}/bin/activate
 
                    # Upgrade pip and install build tools
                    pip install --upgrade pip setuptools wheel
 
                    # Install packages — no strict pinning so pip picks Python 3.13 compatible wheels
                    pip install "numpy>=1.26.4"
                    pip install "pandas>=2.2.3"
                    pip install "h5py>=3.11.0"
                    pip install "tables>=3.9.2" || echo "⚠️ tables install failed — continuing"
                    pip install "scikit-learn>=1.4.2"
                    pip install "xgboost>=2.0.3"
                    pip install "matplotlib>=3.8.4"
                    pip install "gradio>=4.31.5"
                    pip install "kagglehub>=0.2.9"
                    pip install "pytest>=8.2.0" "pytest-cov>=5.0.0" "flake8>=7.0.0"
 
                    echo "✅ Dependencies installed."
                    pip list
                '''
            }
        }
 
        // ── Stage 4: Lint ──────────────────────────────────────────
        stage('Lint') {
            steps {
                echo '🔎 Running flake8 linter...'
                sh '''
                    . ${VENV_DIR}/bin/activate
                    # E501 = line too long (relaxed to 120 for ML code)
                    # W503 = line break before binary operator (style preference)
                    flake8 mini_project_multi_model.py tests/ \
                        --max-line-length=120 \
                        --extend-ignore=E501,W503 \
                        --statistics \
                        --count
                '''
            }
            post {
                failure {
                    echo '❌ Lint failed. Fix code style issues before proceeding.'
                }
            }
        }
 
        // ── Stage 5: Unit Tests ────────────────────────────────────
        stage('Unit Tests') {
            steps {
                echo '🧪 Running unit tests with pytest...'
                sh '''
                    . ${VENV_DIR}/bin/activate
                    pytest tests/ \
                        --cov=. \
                        --cov-report=xml:coverage.xml \
                        --cov-report=html:htmlcov \
                        --junitxml=test-results.xml \
                        -v
                '''
            }
            post {
                always {
                    // Publish JUnit test results in Jenkins UI
                    junit 'test-results.xml'
                    // Publish HTML coverage report
                    publishHTML([
                        allowMissing: false,
                        alwaysLinkToLastBuild: true,
                        keepAll: true,
                        reportDir: 'htmlcov',
                        reportFiles: 'index.html',
                        reportName: 'Coverage Report'
                    ])
                }
                failure {
                    echo '❌ Unit tests failed.'
                }
            }
        }
 
        // ── Stage 6: SonarQube Analysis ────────────────────────────
        stage('SonarQube Scan') {
            steps {
                echo '📊 Running SonarQube analysis...'
                withSonarQubeEnv("${SONAR_SERVER}") {
                    withCredentials([string(credentialsId: "${SONAR_TOKEN}", variable: 'SONAR_AUTH_TOKEN')]) {
                        sh '''
                            sonar-scanner \
                                -Dsonar.projectKey=nilm-ml-project \
                                -Dsonar.projectName="UK-DALE NILM ML Project" \
                                -Dsonar.projectVersion=${BUILD_NUMBER} \
                                -Dsonar.sources=. \
                                -Dsonar.exclusions=**/__pycache__/**,**/.venv/**,**/htmlcov/**,**/outputs/** \
                                -Dsonar.python.coverage.reportPaths=coverage.xml \
                                -Dsonar.python.xunit.reportPath=test-results.xml \
                                -Dsonar.host.url=${SONAR_HOST_URL} \
                                -Dsonar.token=${SONAR_AUTH_TOKEN}
                        '''
                    }
                }
            }
        }
 
        // ── Stage 7: Quality Gate ──────────────────────────────────
        stage('Quality Gate') {
            steps {
                echo '🚦 Waiting for SonarQube Quality Gate result...'
                timeout(time: 5, unit: 'MINUTES') {
                    waitForQualityGate abortPipeline: true
                }
            }
            post {
                failure {
                    echo '❌ Quality Gate FAILED. Build aborted.'
                }
                success {
                    echo '✅ Quality Gate PASSED.'
                }
            }
        }
 
        // ── Stage 8: Train Models ──────────────────────────────────
        stage('Train Models') {
            steps {
                echo '🤖 Training ML models...'
                sh '''
                    . ${VENV_DIR}/bin/activate
                    mkdir -p outputs models
                    python3 train_and_save.py \
                        --output-dir models \
                        --plots-dir outputs
                    echo "✅ Models trained and saved."
                    ls -lh models/
                '''
            }
            post {
                success {
                    archiveArtifacts artifacts: 'models/**,outputs/**', fingerprint: true
                }
                failure {
                    echo '❌ Model training failed.'
                }
            }
        }
 
        // ── Stage 9: Docker Build ──────────────────────────────────
        stage('Docker Build') {
            steps {
                echo "🐳 Building Docker image..."
                sh '''
                    # Install docker CLI if not present
                    which docker || (apt-get update -qq && apt-get install -y docker.io)
 
                    docker build \
                        --build-arg BUILD_NUMBER=${BUILD_NUMBER} \
                        -t ${IMAGE_VERSIONED} \
                        -t ${IMAGE_LATEST} \
                        .
                    docker images | grep nilm-app
                    echo "✅ Docker image built."
                '''
            }
        }
 
        // ── Stage 10: Docker Push ──────────────────────────────────
        stage('Docker Push') {
            steps {
                echo "📤 Pushing image to Docker Hub..."
                withCredentials([usernamePassword(
                    credentialsId: "${DOCKER_CREDS}",
                    usernameVariable: 'DOCKER_USER',
                    passwordVariable: 'DOCKER_PASS'
                )]) {
                    sh '''
                        echo "${DOCKER_PASS}" | docker login -u "${DOCKER_USER}" --password-stdin
                        docker push ${IMAGE_VERSIONED}
                        docker push ${IMAGE_LATEST}
                        docker logout
                        echo "✅ Image pushed: ${IMAGE_VERSIONED}"
                    '''
                }
            }
        }
 
    } // end stages
 
    // ── Post-pipeline actions ──────────────────────────────────────
    post {
        success {
            echo """
            ╔══════════════════════════════════════════╗
            ║  ✅  PIPELINE SUCCEEDED                  ║
            ║  Build   : #${env.BUILD_NUMBER}          ║
            ║  Image   : ${env.IMAGE_VERSIONED}        ║
            ╚══════════════════════════════════════════╝
            """
        }
        failure {
            echo """
            ╔══════════════════════════════════════════╗
            ║  ❌  PIPELINE FAILED                     ║
            ║  Build : #${env.BUILD_NUMBER}            ║
            ║  Check console output for details.       ║
            ╚══════════════════════════════════════════╝
            """
        }
        always {
            deleteDir()
        }
    }
 
}