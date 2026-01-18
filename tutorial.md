# KubeRay Tutorial

## Current architecture

![images/architecture.png](images/architecture.png)
## Set up a Ray cluster

The cluster already runs on lab server `xsel02`. Use the steps below only when rebuilding from scratch.

**Step 1: Create containers**

Creates a default cluster with one worker that can access all GPUs. We use [nvkind](https://github.com/NVIDIA/nvkind) for GPU-enabled kind clusters.

```shell
nvkind cluster create
```
 
**Step 2: Check your clusters**

```shell
nvkind cluster list
nvkind-pl6w2
```

Print GPUs across all nodes (add `--name` to target a specific cluster, or use the current kubecontext):

```shell
nvkind cluster print-gpus
[
    {
        "node": "nvkind-pl6w2-worker",
        "gpus": [
            {
                "Index": "0",
                "Name": "NVIDIA RTX A6000",
                "UUID": "GPU-f4ee8509-da2f-ccf7-1753-02f1eeba984c"
            },
            {
                "Index": "1",
                "Name": "NVIDIA RTX A6000",
                "UUID": "GPU-19fc8822-85b5-4cc4-1327-b68b57f7691d"
            }
        ]
    }
]
```

**Step 3: Launch KubeRay operator**

The operator links Ray clusters to Kubernetes.

**Step 3-1: Add the kuberay repo (already added on lab server)**

```shell
helm repo add kuberay https://ray-project.github.io/kuberay-helm/
helm repo update
```

**Step 3-2: Install both CRDs and KubeRay operator**

```shell
helm install kuberay-operator kuberay/kuberay-operator --version 1.5.1
```

**Step 3-3: Validate installation**

```shell
kubectl get pods

NAME                                READY   STATUS    RESTARTS   AGE
kuberay-operator-6bc45dd644-gwtqv   1/1     Running   0          24s
```

**Step 4: Launch GPU operator**

The GPU Operator makes GPUs schedulable as `nvidia.com/gpu` resources.

**Step 4-1: Add the NVIDIA Helm repository**  

```shell
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia \
    && helm repo update
```

**Step 4-2: Deploy GPU Operator**

```shell
helm install --wait --generate-name \
    -n gpu-operator --create-namespace \
    nvidia/gpu-operator
```

Allow 5–10 minutes for the operator and operands to become ready.

**Step 5: Set up Ray pods**

**Step 5-0: Add HF token and build pods**

Before applying, edit `ray-cluster.yaml` with your Hugging Face token.

```
vim ray-cluster.yaml

apiVersion: v1
kind: Secret
metadata:
  name: hf-token
type: Opaque
stringData:
  hf_token: <your token>
```

Then apply the manifest:

```shell
kubectl apply -f ray-cluster.yaml
```

**Step 5-1: Check the running pods**

Ensure each pod shows `READY 1/1`, indicating clean initialization.

```shell
kubectl get pods

NAME                                   READY   STATUS    RESTARTS   AGE
kuberay-operator-648fdf69cb-4wpkt      1/1     Running   0          2d22h
llm-cluster-gpu-workers-worker-2n8ft   1/1     Running   0          47h
llm-cluster-gpu-workers-worker-j9k9s   1/1     Running   0          47h
llm-cluster-head-v29mn                 1/1     Running   0          47h

kubectl get pods -o wide # you can see what container is used
NAME                                   READY   STATUS    RESTARTS   AGE     IP            NODE                  NOMINATED NODE   READINESS GATES
kuberay-operator-648fdf69cb-4wpkt      1/1     Running   0          2d22h   10.244.1.11   nvkind-pl6w2-worker   <none>           <none>
llm-cluster-gpu-workers-worker-2n8ft   1/1     Running   0          47h     10.244.1.45   nvkind-pl6w2-worker   <none>           <none>
llm-cluster-gpu-workers-worker-j9k9s   1/1     Running   0          47h     10.244.1.46   nvkind-pl6w2-worker   <none>           <none>
llm-cluster-head-v29mn                 1/1     Running   0          47h     10.244.1.44   nvkind-pl6w2-worker   <none>           <none>
```

**Step 5-2 (optional): Check nodes**

```shell
kubectl get node

NAME                         STATUS   ROLES           AGE     VERSION
nvkind-pl6w2-control-plane   Ready    control-plane   2d22h   v1.35.0
nvkind-pl6w2-worker          Ready    <none>          2d22h   v1.35.0
```

**Step 5-3 (optional): Describe nodes for detail**

```shell
kubectl describe node

Name:               nvkind-pl6w2-control-plane
Roles:              control-plane
Labels:             beta.kubernetes.io/arch=amd64
                    beta.kubernetes.io/os=linux
...
```

## Submit a Ray training job

**Step 1: Access the head node**

```shell
kubectl exec --stdin --tty <your-head-node> -- /bin/bash
or
kubectl exec -i -t <your-head-node> -- /bin/bash

# Our case
kubectl exec --stdin --tty llm-cluster-head-v29mn -- /bin/bash
```

**Step 2: Copy pyproject.toml**

Run this inside **the head node**

```shell
pwd
/home/ray

ls
anaconda3  pip-freeze.txt  rayllm_py311_cu128.lock  requirements_compiled.txt  training  workspace

cp training/pyproject.toml workspace/
```

**Step 3: Run a training script**

```shell
ray job submit --runtime-env-json '{"working_dir": ".", "py_executable": "uv run"}' -- uv run src/train_llm.py

Job submission server address: http://10.244.1.44:8265
2026-01-17 17:48:23,225 INFO dashboard_sdk.py:355 -- Uploading package gcs://_ray_pkg_3383f23f6a634a99.zip.
2026-01-17 17:48:23,225 INFO packaging.py:588 -- Creating a file package for local module '.'.
```

`uv` keeps dependencies consistent across pods (e.g., workers). Details to be expanded soon.

- https://www.anyscale.com/blog/uv-ray-pain-free-python-dependencies-in-clusters

**Step 4: Check training or pod status via port-forward**

```shell
kubectl port-forward <your-head-node> 8265:8265

# Our case
kubectl port-forward svc/llm-cluster-head-svc 8265:8265
```

In VS Code, open the Ray dashboard after port-forwarding. Example view:

![image](images/vscode-port.png)

**(Optional): Copy files or directories from the local to the pod**

```shell
kubectl cp <file or dir> <head-node>:/home/ray/
```

## Checkpoints

The cluster uses PVC-backed shared storage at `/mnt/shared`, defined in `ray-cluster.yaml`.

```shell
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: ray-checkpoints-pvc
  namespace: default
spec:
  storageClassName: standard
  accessModes:
    - ReadWriteOnce  # Rank 0 only - local-path provisioner supports this
  resources:
    requests:
      storage: 100Gi
```

## Serving (TBD)

Build and serve the LLM application:

```shell
serve build llm-serve.yaml
```

Currently blocked by an intermittent vLLM error.

## Termination

**Step 1: Remove Ray pods**

```shell
kubectl delete raycluster llm-cluster 
```

- `raycluster`: CRD (Custom Resource Definition) in k8s
- `llm-cluster`: Ray cluster name defined in the YAML

**Step 2: Remove kind clusters**

```shell
for cluster in $(kind get clusters); do kind delete cluster --name=${cluster}; done
```

## Inside the head node

```shell
$HOME = /home/ray
$TMP = /tmp/ray (This variable is not actually set)
```

## Troubleshooting
### GPUs not available

Docker may lack GPU configuration if you see a message like:

```bash
message: 
'Deployment ''LLMServer:qwen2_5-7b-instruct'' in application
''llms'' has 1 replicas that have taken more than 30s to be scheduled. This may be due to waiting for the cluster to auto-scale or for a runtime environment to be installed. Resources required for each replica: [{"CPU": 1.0, "GPU": 1.0}], total resources available: {}. Use `ray status` for more details.'
```

Then check available resources:

```bash
kubectl get nodes -o json | grep -A 10 "allocatable"

"**allocatable**": 
{
	"cpu": "48",
	"ephemeral-storage": "225956404Ki",
	"hugepages-1Gi": "0",
	"hugepages-2Mi": "0",
	"memory": "131758472Ki",
	"pods": "110"
},
```

If host GPUs exist but are absent in logs, install the NVIDIA device plugin (`gpu-operator`) so GPUs appear as `nvidia.com/gpu`.

> Note: `nvidia-ctk` 1.18.1 is unstable; prefer 1.17.1.

```shell
# This fails to umount
VERSION:
   1.18.1
commit: efe99418ef87500dbe059cadc9ab418b2815b9d5

# This works
VERSION:
   1.17.1
commit: 1467f3f339a855f20d179e9883e418a09118a93e
```
