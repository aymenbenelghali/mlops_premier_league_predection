Kubernetes manifests for pl-predictor

Quick notes
- The container in the project's `Dockerfile` exposes port `5000` — manifests target that port.
- The Deployment uses the image `pl-predictor:latest`. Change this to your registry image if you push the image.
- `Service` is type `LoadBalancer` by default; for local clusters change it to `NodePort` or use `kubectl port-forward`.

Apply the manifests

1. (Optional) If you're using a local cluster and built the image locally:

  - For minikube:

    minikube image load pl-predictor:latest

  - For kind (after building the image):

    kind load docker-image pl-predictor:latest

  - Alternatively set `imagePullPolicy: Never` in the Deployment and ensure the node can see your local Docker images.

2. Apply manifests:

  kubectl apply -f k8s/deployment.yaml
  kubectl apply -f k8s/service.yaml
  kubectl apply -f k8s/ingress.yaml

3. Access the app

- If your Service becomes a `LoadBalancer` with a public IP (cloud): open its external IP.
- For local testing you can port-forward:

  kubectl port-forward deployment/pl-predictor 5000:5000

  Then open http://localhost:5000

Notes about purpose

- Kubernetes is an orchestration platform that helps run, scale, and manage containerized applications across a cluster of machines.
- The manifests here let you run multiple replicas of your predictor, provide a stable network endpoint (`Service`), and optionally route HTTP traffic via an `Ingress`.
- Use Kubernetes when you need high availability, scaling, rolling updates, or to run in managed cloud environments (EKS/GKE/AKS).
