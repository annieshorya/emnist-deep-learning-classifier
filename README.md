# Handwritten Character Recognition: EMNIST (62 classes)

Three neural-network architectures compared on **EMNIST ByClass**: 62 classes (digits 0–9 plus
upper- and lower-case letters), 100,000 training and 20,000 test images of 28×28 handwritten
characters.

**Result:** the CNN reached **86.0% test accuracy** after Hyperband tuning, ahead of LeNet-5
(84.3%) and a multi-layer perceptron (82.8%).

| Model | Framework | Test accuracy | Notes |
|---|---|---|---|
| **CNN** (4 conv blocks, dropout) | TensorFlow / Keras | 85.6% → **86.0%** after tuning | Hyperband (Keras Tuner) over filters, dropout and learning rate |
| **LeNet-5** | PyTorch | 84.3% | Grid search (skorch + GridSearchCV) over learning rate / optimiser; fastest to train |
| **MLP** (3 hidden layers: 512-256-128) | TensorFlow / Keras | **82.8%** untuned, 78.3% after tuning | GridSearchCV (scikeras) picked a configuration that generalised worse |

## What I took away

- **Convolutions matter for images.** The CNN beat the fully connected MLP by about 3 points,
  because it learns local stroke patterns instead of treating pixels independently.
- **Tuning isn't free.** Hyperband helped the CNN slightly (+0.5 points), but the grid-searched
  MLP *lost* 4.5 points of accuracy on the test set. It chose the configuration with the best
  cross-validation score, which overfit. A held-out check after tuning is not optional.
- **Accuracy hides the hard classes.** For LeNet-5, macro-F1 was 0.69 against 0.84 accuracy.
  The best classes (`3`, `7`, `8`, `e`) score F1 above 0.95. Lower-case letters that look the same as
  their capitals (`o`, `s`, `c`, `m`, `u`, `p`, `v`, `y`), and `l` against `1`/`I`, score close
  to zero, because a lone character gives no case information.

## What I'd do next

- Wire the augmentation into CNN training. An `ImageDataGenerator` is set up in the notebook,
  but `model.fit` trains on the raw arrays. I would also drop horizontal flips and 90° rotations,
  since a mirrored `b` is a `d`.
- Merge or re-weight the case-ambiguous classes, or report accuracy on the 47-class *Balanced* split as well.

## Approach

- **Preprocessing:** pixel normalisation, a channel dimension for the CNN, reshaping and
  random-rotation augmentation for LeNet-5 (`torchvision.transforms`).
- **Training:** early stopping and learning-rate reduction on plateau.
- **Tuning:** Hyperband (CNN), GridSearchCV with 3-fold CV (MLP via scikeras, LeNet-5 via skorch).
- **Evaluation:** accuracy, weighted precision / recall / F1, per-class classification reports
  and confusion matrices on the untouched test set.

## Run it

`notebook.ipynb` contains the full pipeline with outputs. It expects `emnist_train.pkl` and
`emnist_test.pkl` (a 100k / 20k sample of EMNIST ByClass supplied with the course) in the
working directory. The same data can be downloaded with
`torchvision.datasets.EMNIST(root, split="byclass", download=True)`.

```bash
pip install -r requirements.txt
jupyter notebook notebook.ipynb
```

## Context

Group assignment for a deep-learning unit, Master of Computer Science, University of Sydney
(2024). Team: Siddharth Jadli, Annie Shorya, Linxiao Li.
