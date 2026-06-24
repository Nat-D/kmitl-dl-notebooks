# Colab notebooks — real PyTorch training on real datasets

Companion notebooks for the **Fundamentals of Deep Learning** course. The in-browser
sandbox runs stdlib-only Python (the *from-scratch* exercises); these notebooks are the
complement — **train the real thing on a real dataset, on a free GPU**. Each opens with a
hand-off from the by-hand work, then follows the same rhythm: setup/GPU-check → get the
data → **look at the data** → build (`🔧`) → train (`🔧`) → evaluate → experiment → reflect.

> Run on a GPU: in Colab, **Runtime → Change runtime type → T4 GPU**.

| Notebook | Lecture | Dataset | What you train |
|---|---|---|---|
| `01-l2-fc-penguins` | L2 — fully-connected nets | Palmer Penguins (tabular) | an MLP on a real CSV you can see |
| `02-l5-train-classifier-fashionmnist` | L5 — training-loop lab | Fashion-MNIST | the 5-step loop; curves, early stop, confusion matrix, fix-the-broken-run |
| `03-l6-cnn-fashionmnist-cifar` | L6 — CNNs | Fashion-MNIST → CIFAR-10 | a CNN that beats the L5 dense net on the same data; then colour images |
| `04-l9-rnn-names` | L9 — RNNs | names → language (PyTorch tutorial set) | an LSTM classifying surnames from the final hidden state |
| `05-l11-sequence-shakespeare` | L11 — sequence lab | tiny-shakespeare | a char-level language model that generates text (temperature, perplexity) |
| `06-capstone-starter` | Capstone | your choice (vision / text / tabular) | your own model, honest held-out test, write-up |

The training loop matures across the set (the same `train / evaluate / plot_curves`
shape), and the datasets climb in realism: tabular → grayscale images → colour →
text-classification → text-generation → free choice.

**Datasets** download at runtime (Colab has internet): vision via `torchvision.datasets`
(`download=True`); the names set from `download.pytorch.org`; tiny-shakespeare from the
`char-rnn` repo. No `pip install` needed — Colab's stock `torch`/`torchvision` are used.

**Status:** notebooks are authored and statically validated (valid JSON, every code cell
parses; several were also run on CPU with tiny/synthetic data). They still need a quick
**GPU smoke-test on Colab** before the "Open in Colab" links go live to students.
