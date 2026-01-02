# Week 1: Introduction to MLOps

## 📚 Overview
This week covered the foundational concepts of MLOps, version control with GitHub, and practical implementation of ML pipelines. The focus was on understanding the gap between traditional ML development and production-ready ML systems.

---

## 📋 Table of Contents
1. [Session 1: Introduction to MLOps](#session-1-introduction-to-mlops)
2. [Session 2: Version Control](#session-2-version-control)
3. [Practical Implementation](#practical-implementation)
4. [Key Takeaways](#key-takeaways)
5. [Resources](#resources)

---

## Session 1: Introduction to MLOps

### 🎯 Reality of AI in the Market
- Understanding the challenges of deploying AI/ML models in production
- The gap between model development and real-world deployment
- Why most ML projects fail to reach production

### 📖 Introduction

#### Standard ML Cycle
The traditional machine learning workflow:
1. **Data Collection** - Gathering relevant datasets
2. **Data Preprocessing** - Cleaning and preparing data
3. **Feature Engineering** - Creating meaningful features
4. **Model Training** - Building and training models
5. **Model Evaluation** - Testing model performance
6. **Deployment** - (Often overlooked in traditional ML)

#### What is DevOps?
- **Definition**: Development + Operations
- **Goal**: Automate and integrate the processes between software development and IT teams
- **Key Principles**:
  - Continuous Integration (CI)
  - Continuous Delivery (CD)
  - Automation
  - Monitoring and Logging
  - Collaboration

#### What is MLOps?
- **Definition**: Machine Learning + Operations
- **Purpose**: Apply DevOps principles to ML systems
- **Key Components**:
  - Version Control (Code, Data, Models)
  - Automated Testing
  - Continuous Training
  - Model Monitoring
  - Reproducibility
  - Scalability

**MLOps = DevOps + Data Science + Machine Learning**

### 🔄 Machine Learning Lifecycle
A comprehensive view of the ML lifecycle in production:

1. **Problem Definition**
   - Business understanding
   - Success metrics definition

2. **Data Engineering**
   - Data collection
   - Data validation
   - Data versioning

3. **Model Development**
   - Feature engineering
   - Model selection
   - Hyperparameter tuning

4. **Model Validation**
   - Performance evaluation
   - A/B testing
   - Bias detection

5. **Model Deployment**
   - Serving infrastructure
   - API development
   - Containerization

6. **Monitoring & Maintenance**
   - Performance tracking
   - Data drift detection
   - Model retraining

### 🔧 Introduction to Version Control

#### Key Aspects of Version Control
- **Track Changes**: Monitor modifications over time
- **Collaboration**: Multiple team members working together
- **Backup**: Prevent data loss
- **Branching**: Parallel development workflows
- **History**: Complete audit trail of changes

#### Types of Version Control Systems

1. **Local Version Control**
   - Single database on local machine
   - Limited collaboration capabilities

2. **Centralized Version Control (CVCS)**
   - Single server contains all versions
   - Examples: SVN, Perforce
   - Limitations: Single point of failure

3. **Distributed Version Control (DVCS)**
   - Every user has complete repository
   - Examples: Git, Mercurial
   - Benefits: Better collaboration, offline work, redundancy

### 📅 Next Two Weeks Plan
- Deep dive into Git and GitHub
- CI/CD pipelines for ML
- Docker and containerization
- Model versioning and tracking

---

## Session 2: Version Control

### 🐙 Using GitHub for Version Control

#### What is GitHub?
- Web-based hosting service for Git repositories
- Collaboration platform for developers
- Features:
  - Code hosting
  - Issue tracking
  - Pull requests
  - Project management
  - GitHub Actions (CI/CD)

#### Setting Up GitHub
1. Create GitHub account
2. Configure Git locally
3. Set up SSH keys (optional but recommended)
4. Configure user name and email

```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

#### Creating a Repository
- Initialize new repository on GitHub
- Choose public or private visibility
- Add README, .gitignore, and license
- Clone to local machine

```bash
git clone https://github.com/username/repository.git
```

#### Cloning a Repository
```bash
# Clone via HTTPS
git clone https://github.com/username/repo.git

# Clone via SSH
git clone git@github.com:username/repo.git
```

#### Making Changes
```bash
# Check status
git status

# Add specific files
git add filename.py

# Add all changes
git add .
```

#### Committing Changes
```bash
# Commit with message
git commit -m "Descriptive commit message"

# Commit with detailed message
git commit -m "Title" -m "Detailed description"
```

**Best Practices for Commit Messages**:
- Use present tense ("Add feature" not "Added feature")
- Be descriptive but concise
- Reference issue numbers when applicable

#### Pushing Changes
```bash
# Push to remote repository
git push origin main

# Push new branch
git push -u origin branch-name
```

#### Branching
Branches allow parallel development without affecting the main codebase.

```bash
# Create new branch
git branch feature-branch

# Switch to branch
git checkout feature-branch

# Create and switch in one command
git checkout -b feature-branch

# List all branches
git branch -a

# Delete branch
git branch -d feature-branch
```

**Branching Strategies**:
- **main/master**: Production-ready code
- **develop**: Integration branch
- **feature/**: New features
- **hotfix/**: Emergency fixes
- **release/**: Release preparation

#### Pull Requests
- Propose changes to repository
- Code review process
- Discussion and collaboration
- Merge after approval

**Pull Request Workflow**:
1. Create feature branch
2. Make changes and commit
3. Push branch to GitHub
4. Open pull request
5. Review and discuss
6. Merge into main branch

#### Collaborating with Others
```bash
# Fetch changes from remote
git fetch origin

# Pull changes (fetch + merge)
git pull origin main

# View remote repositories
git remote -v

# Add collaborators on GitHub
# Settings → Collaborators → Add people
```

### 🔄 Revisiting ML Cycle

#### ML Pipeline Example
Understanding how version control fits into ML workflows:

- **Code Versioning**: Track model code, training scripts
- **Data Versioning**: Track dataset changes (using DVC, Git LFS)
- **Model Versioning**: Track trained models and metrics
- **Experiment Tracking**: Document hyperparameters and results

### 🏭 Industry Trivia
- **87% of ML projects never make it to production** (VentureBeat)
- **Only 22% of companies using ML have successfully deployed a model** (Gartner)
- **MLOps can reduce model deployment time from months to days**
- **Version control is the foundation of reproducible ML**

---

## Practical Implementation

### 🔬 ML Pipeline Example

This week included a hands-on implementation of a complete ML pipeline using the Iris dataset.

#### Technologies Used
```python
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from hyperopt import fmin, tpe, hp
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
```

#### Pipeline Components

1. **Data Loading**
```python
def read_csv(file_path):
    return pd.read_csv(file_path)
```

2. **Feature Engineering**
```python
def create_features(data):
    # Feature creation logic
    return data
```

3. **Model Training**
```python
def train_classifier(data):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, accuracy
```

4. **Hyperparameter Tuning with Hyperopt**
```python
def objective(params):
    model = RandomForestClassifier(**params)
    score = cross_val_score(model, X, y, cv=5).mean()
    return -score  # Minimize negative accuracy

# Define search space
space = {
    'n_estimators': hp.choice('n_estimators', range(10, 101)),
    'max_depth': hp.choice('max_depth', range(1, 21))
}

# Run optimization
best_params = fmin(
    fn=objective, 
    space=space, 
    algo=tpe.suggest, 
    max_evals=100
)
```

5. **Model Evaluation**
```python
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy
```

#### Complete Pipeline Implementation
```python
# Define preprocessing and model pipeline
pipeline = Pipeline([
    ('preprocessor', ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), X.columns)
        ],
        remainder='passthrough'
    )),
    ('classifier', RandomForestClassifier())
])

# Train the model
pipeline.fit(X_train, y_train)

# Evaluate
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy on test set: {accuracy}")
```

#### Results
- **Model Accuracy**: 100% on test set (Iris dataset)
- **Best Hyperparameters**: 
  - `n_estimators`: 45
  - `max_depth`: 16
- **Optimization**: Used Tree of Parzen Estimators (TPE) algorithm
- **Cross-validation**: 5-fold CV for robust evaluation

#### Key Learnings from Implementation
1. **Pipeline Benefits**:
   - Reproducibility
   - Prevents data leakage
   - Simplifies deployment
   - Easy to version control

2. **Hyperparameter Tuning**:
   - Automated search vs. manual tuning
   - TPE is more efficient than grid search
   - Cross-validation prevents overfitting

3. **Version Control Integration**:
   - Track code changes
   - Document experiments
   - Collaborate with team members

---

## Key Takeaways

### 🎓 Concepts Learned

1. **MLOps Fundamentals**
   - Bridge between ML development and production
   - Importance of automation and monitoring
   - Lifecycle management of ML models

2. **Version Control Mastery**
   - Git basics and advanced workflows
   - GitHub collaboration features
   - Branching strategies for ML projects

3. **ML Pipeline Design**
   - Modular code structure
   - Preprocessing automation
   - Hyperparameter optimization
   - Model evaluation best practices

4. **Industry Awareness**
   - Real-world challenges in ML deployment
   - Importance of reproducibility
   - Team collaboration in ML projects

### 🛠️ Skills Acquired

- ✅ Git and GitHub proficiency
- ✅ Creating reproducible ML pipelines
- ✅ Hyperparameter tuning with Hyperopt
- ✅ Scikit-learn Pipeline API
- ✅ Collaborative development workflows
- ✅ Understanding MLOps principles

### 📝 Best Practices

1. **Version Control**
   - Commit frequently with meaningful messages
   - Use branches for new features
   - Review code through pull requests
   - Keep repositories organized

2. **ML Development**
   - Use pipelines for reproducibility
   - Document experiments and results
   - Version datasets and models
   - Automate repetitive tasks

3. **Collaboration**
   - Clear communication in commits
   - Code reviews for quality
   - Documentation for team members
   - Issue tracking for tasks

---

## Resources

### 📚 Documentation
- [Git Documentation](https://git-scm.com/doc)
- [GitHub Guides](https://guides.github.com/)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Hyperopt Documentation](http://hyperopt.github.io/hyperopt/)

### 🎥 Learning Materials
- [Git and GitHub for Beginners](https://www.youtube.com/watch?v=RGOj5yH7evk)
- [MLOps Explained](https://ml-ops.org/)
- [Scikit-learn Pipelines](https://scikit-learn.org/stable/modules/compose.html)

### 🔗 Useful Links
- [GitHub Learning Lab](https://lab.github.com/)
- [MLOps Community](https://mlops.community/)
- [Awesome MLOps](https://github.com/visenger/awesome-mlops)

---

## 📁 Repository Structure

```
Week_1/
├── README.md                          # This file
├── Week_1_ML_pipeline_example.ipynb   # ML Pipeline implementation
├── Iris.csv                           # Dataset
└── .git/                              # Git repository
```

---

## 🎯 Next Steps

### Week 2 Preview
- Advanced Git workflows
- CI/CD for ML projects
- Docker containerization
- Model deployment strategies
- Experiment tracking tools

### Practice Exercises
1. Create a GitHub repository for a personal ML project
2. Implement a complete ML pipeline with version control
3. Practice branching and pull request workflows
4. Experiment with different hyperparameter tuning methods
5. Document your experiments in README files

---

## 📊 Progress Tracker

- [x] Understanding MLOps concepts
- [x] Git and GitHub basics
- [x] Creating repositories
- [x] Branching and merging
- [x] Pull requests
- [x] ML Pipeline implementation
- [x] Hyperparameter tuning
- [x] Model evaluation
- [ ] CI/CD pipelines (Week 2)
- [ ] Docker containerization (Week 2)

---

## 🤝 Contributing

This is a personal learning repository. However, suggestions and improvements are welcome!

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -m 'Add improvement'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Open a Pull Request

---

## 📧 Contact

For questions or discussions about MLOps and this learning journey, feel free to reach out!

---

## 📄 License

This project is for educational purposes.

---

**Week 1 Completed**: ✅ January 2, 2026

*"The journey of a thousand miles begins with a single commit."* 🚀
