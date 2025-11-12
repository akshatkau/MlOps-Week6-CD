# MLOps Week 6 & 7 – Continuous Deployment, Autoscaling, and Observability

## Overview
This project extends the Iris FastAPI Machine Learning API into a complete MLOps pipeline with automated deployment, autoscaling, and observability on Google Cloud Platform (GCP). It uses Docker, GitHub Actions, Google Artifact Registry, and Google Kubernetes Engine (GKE).

---

## Week 6 – Continuous Deployment

In Week 6, Continuous Deployment (CD) was implemented on top of the Continuous Integration (CI) workflow.

### Key Components
- **Docker**: Containerized the Iris FastAPI model.
- **GitHub Actions**:
  - **CI Workflow**: Runs linting and testing using `flake8`.
  - **CD Workflow**: Builds the Docker image, pushes it to Google Artifact Registry, and deploys to GKE.
- **Kubernetes Deployment**:
  - Defined `deployment.yaml` and `service.yaml` to run and expose the model as a REST API on a GKE cluster.

**Outcome:** Successful end-to-end CI/CD pipeline deploying the Iris model API to a GKE cluster.

---

## Week 7 – Autoscaling, Observability, and Stress Testing

In Week 7, the system was enhanced for scalability and monitoring to handle production-level workloads.

### Enhancements
- **Autoscaling with HPA (Horizontal Pod Autoscaler)**:
  - Configured minimum 1 and maximum 3 replicas with a target CPU utilization of 50%.
  - Demonstrated automatic scaling from 1 to 3 pods under load.
- **Observability**:
  - Integrated OpenTelemetry for distributed tracing.
  - Implemented structured JSON logging exported to Google Cloud Logging and Trace.
  - Logs include latency, input features, predictions, and trace IDs.
- **Stress Testing**:
  - Used `wrk` to simulate high concurrency:
    ```bash
    wrk -t4 -c1000 -d30s --latency -s post.lua http://<EXTERNAL-IP>:80/predict
    ```
  - Observed automatic scaling and load distribution among pods.

**Outcome:** The system demonstrated elastic scaling, improved resilience, and complete observability under high traffic.
