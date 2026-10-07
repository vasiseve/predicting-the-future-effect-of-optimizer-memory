# Data and experimental protocol

Public dataset sources (downloaded at runtime, not redistributed):

- CIFAR-10: https://huggingface.co/datasets/uoft-cs/cifar10
- Tiny ImageNet: https://huggingface.co/datasets/zh-plus/tiny-imagenet

Consult the dataset cards and original dataset terms for attribution and usage. Dependencies retain their upstream licenses. No new open-source license or copyright owner has been assigned by this code package preparation.

CIFAR-10 tasks use classes 0–4 and 5–9; Tiny ImageNet tasks use classes 0–99 and 100–199. Labels are remapped within each task. The original loader takes the first requested number of samples per class in dataset order. Full-grid settings use 1,000 training / 500 test examples per CIFAR class and 500 training / 50 validation examples per Tiny ImageNet class. Training uses random crop and horizontal flip; evaluation is deterministic. Exact normalization values and split selection are in `rttp_grid/data.py`.

Boundary training uses batch size 128; epochs are 12/20 for CIFAR SmallCNN/ResNet-18 and 10/15 for Tiny ImageNet. Boundary SGD learning rates are 0.03/0.01 and boundary Adam learning rates are 0.001/0.0003 for CIFAR/Tiny ImageNet. Optimizer-specific response learning rates and perturbation sweeps are separate and recorded in `configs/full_grid.json`.

Continuation batches are materialized once and shared by nominal, tangent and finite-difference counterfactual trajectories. If fewer batches are available than the maximum horizon, materialized batches are repeated. Continuation uses fixed model buffers; inspect `configure_functional_model` and `FunctionalModel` for BatchNorm behavior. The paper experiment uses float32; offline derivative checks use float64 to resolve numerical errors.

The supplied historical environment recorded Python 3.12.13, PyTorch 2.11.0+cu128, torchvision 0.26.0+cu128 and CUDA execution on Linux. GPU model, full runtime and exact dataset revisions were absent. This archive pins the recorded direct dependency versions, while preserving scientific settings. Full dependency resolution and new hardware may change numerical results.

For large runs, disk stores boundary checkpoints and chunked response tables; GPU memory grows with model and number of tangent directions. This archive has no measured minimum hardware requirement. Run the small real-data smoke configuration before allocating a full-grid run. Timing outputs synchronize CUDA, and GPU-memory outputs are unavailable (`NaN`) on CPU.
