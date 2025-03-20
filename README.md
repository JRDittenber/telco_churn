# Customer Churn Prediction: End-to-End Data Science Project

## Overview
This project is a comprehensive end-to-end data science solution for predicting customer churn in a subscription-based business. It includes data pipeline creation, model training, deployment, and real-time predictions, simulating an industry-standard workflow.

## Project Architecture
1. **Data Collection & Pipeline**:
   - **External Data Source**: Simulated data collection using web scraping/APIs.
   - **ETL Process**: Data extracted, cleaned, and stored using Python scripts and managed with Apache Airflow.
   - **Storage**: Data stored in a SQL database/AWS S3 bucket for persistence.

2. **Exploratory Data Analysis (EDA)**:
   - Visualization of data distributions and correlations.
   - Identification of key features influencing churn.

3. **Model Development**:
   - Initial model selection (Logistic Regression, Decision Trees).
   - Advanced modeling with Random Forest and XGBoost.
   - Hyperparameter tuning via GridSearchCV.
   - Cross-validation for robust model evaluation.

4. **Deployment**:
   - **Model API**: Built with Flask/FastAPI for serving real-time predictions.
   - **Containerization**: Docker used to ensure consistent deployment.
   - **Cloud Deployment**: Hosted on AWS EC2 for scalable access.

5. **Monitoring**:
   - Real-time model performance tracking with Prometheus and Grafana.
   - Automated retraining triggered by new data pipelines.

## Tools and Technologies
- **Languages**: Python, SQL
- **Libraries**: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `matplotlib`, `seaborn`, `shap`
- **ETL and Orchestration**: Apache Airflow/Prefect
- **Deployment**: Flask/FastAPI, Docker, AWS (EC2, S3)
- **Visualization**: Streamlit (optional interactive dashboard)
- **Monitoring**: Prometheus, Grafana

## Project Workflow
### 1. Data Collection
- **Script**: Python scripts extract data from an API or web source.
- **Automation**: Scheduled with Apache Airflow for daily data ingestion.
- **ETL Steps**: Data cleaned, transformed, and stored in SQL/AWS S3.

### 2. EDA and Feature Engineering
- Detailed analysis of data distribution and feature relationships.
- Generation of new features for better predictive power.
- Visual exploration with `matplotlib` and `seaborn`.

### 3. Model Training
- **Initial Training**: Baseline model trained with `LogisticRegression`.
- **Advanced Models**: Experimentation with `RandomForest` and `XGBoost`.
- **Tuning**: Hyperparameters optimized using `GridSearchCV`.
- **Cross-Validation**: k-fold cross-validation to ensure model reliability.

### 4. Deployment
- **API Creation**: Flask/FastAPI for real-time model predictions.
- **Containerization**: Docker used for deploying the API.
- **Cloud Setup**: Deployed to AWS EC2 instance for public access.

### 5. Monitoring and Updates
- **Performance Metrics**: Monitored with Prometheus and visualized using Grafana.
- **Retraining**: Pipeline configured to retrain models automatically when new data is added.

## Installation and Setup
### Prerequisites
- Python 3.8+
- Docker
- GitHub Actions
- MongoDB account (for simulating data ingestion)
- AWS account (optional for cloud deployment)

### Steps to Run Locally
1. **Clone the Repository**:
    ```bash
    git clone https://github.com/yourusername/churn-prediction.git
    cd churn-prediction
    ```

2. **Set Up Virtual Environment**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Run ETL Pipeline**:
    - Use Apache Airflow/Prefect to schedule and execute the pipeline.

5. **Train the Model**:
    ```bash
    python train_model.py
    ```

6. **Start the API**:
    ```bash
    python app.py
    ```

7. **Run Docker Container**:
    ```bash
    docker build -t churn-prediction-api .
    docker run -p 5000:5000 churn-prediction-api
    ```

### Deployment on AWS EC2
- Follow [this guide](https://docs.aws.amazon.com/ec2/index.html) to deploy your containerized app on an EC2 instance.

## Future Enhancements
- **Interactive Dashboard**: Implement Streamlit for user interaction.
- **Continuous Integration/Deployment**: Add GitHub Actions/Jenkins for automated updates.
- **Additional Data Sources**: Integrate more external data for enriched analysis.

## License
This project is licensed under the MIT License.

## Contact
For any questions or collaboration requests, feel free to reach out:
- **Email**: [jeff.dittenber@hotmail.com](mailto:jeff.dittenber@hotmail.com)
- **LinkedIn**: [linkedin.com/in/passionformath](https://www.linkedin.com/in/passionformath/)
