
# MLOps ALPR Pipeline

This project implements an end-to-end MLOps pipeline for Automatic License Plate Recognition (ALPR) using YOLOv8 for object detection, Flask for serving the model, and Kubernetes for orchestration. The pipeline leverages GitHub Actions for Continuous Integration and Continuous Deployment (CI/CD), ensuring automated testing and deployment.

## Features

- **YOLOv8 Integration**: Utilizes the latest YOLOv8 model for accurate and efficient license plate detection.
- **Flask API**: Provides a RESTful API endpoint for processing images and returning detected license plates.
- **Kubernetes Deployment**: Ensures scalable and reliable deployment using Kubernetes manifests.
- **CI/CD with GitHub Actions**: Automates testing, building, and deployment processes.
- **Dockerized Application**: Encapsulates the application in a Docker container for consistent environments across development and production.

## Project Structure

```
mlops-alpr-pipeline/
├── .github/workflows/    # CI/CD workflows
├── k8s/                  # Kubernetes deployment manifests
├── src/                  # Source code for the Flask application
├── tests/                # Unit and integration tests
├── Dockerfile            # Dockerfile for building the application image
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

## Getting Started

### Prerequisites

- Docker installed on your machine
- Python 3.8 or higher
- Access to a Kubernetes cluster (e.g., Minikube, GKE, EKS)

### Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/christy136/mlops-alpr-pipeline.git
   cd mlops-alpr-pipeline
   ```

2. **Install Python dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Build the Docker image**:

   ```bash
   docker build -t alpr-app .
   ```

4. **Run the application locally**:

   ```bash
   docker run -p 5000:5000 alpr-app
   ```

   The API will be accessible at `http://localhost:5000`.

## Usage

Send a POST request to the `/predict` endpoint with an image file:

```bash
curl -X POST -F 'image=@path_to_image.jpg' http://localhost:5000/predict
```

The response will contain the detected license plate numbers and their bounding box coordinates.

## Kubernetes Deployment

1. **Apply Kubernetes manifests**:

   ```bash
   kubectl apply -f k8s/
   ```

2. **Verify the deployment**:

   ```bash
   kubectl get pods
   ```

3. **Access the service**:

   Depending on your Kubernetes setup, you can access the service via `NodePort`, `LoadBalancer`, or port-forwarding.

## CI/CD with GitHub Actions

The project includes GitHub Actions workflows located in `.github/workflows/` that automate the following:

- Running tests on pull requests and pushes to the `main` branch.
- Building and pushing Docker images to a container registry.
- Deploying the application to a Kubernetes cluster.

Ensure you have the necessary secrets configured in your GitHub repository settings for authentication with your container registry and Kubernetes cluster.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

For more information and updates, visit the [GitHub repository](https://github.com/christy136/mlops-alpr-pipeline).
