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
| `07-l10-vanishing-gradients-lstm` | L10 — vanishing gradients & LSTMs | *(no dataset)* | see the gradient **vanish** through a `tanh` RNN and **survive** the LSTM cell-state highway; gates store/hold/overwrite; `nn.LSTM`/`nn.GRU` + the 4× cost; gradient clipping |
| `05-l11-sequence-shakespeare` | L11 — sequence lab | tiny-shakespeare | a char-level language model that generates text (temperature, perplexity) |
| `08-l12-encoder-decoder` | L12 — encoder-decoder & representation learning | MNIST | an autoencoder; a bottleneck forcing a compressed representation; latents as embeddings (cosine nearest-neighbours); a frozen-encoder linear probe vs raw pixels |
| `09-l13-seq2seq-attention` | L13 — sequence-to-sequence & attention | generated digit sequences; no downloads | understand a weighted blend first, then train GRU models with/without attention and inspect learned alignment |
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

## Lecture 13 — attention, one step at a time

[Open the notebook in Colab](https://colab.research.google.com/github/Nat-D/kmitl-dl-notebooks/blob/main/09-l13-seq2seq-attention.ipynb).

- **Part 0:** build the plain encoder–decoder, pass the final state to the decoder, train with teacher forcing, and generate until EOS. Start here, before attention.
- **Part A:** given weights → weighted sum → query/key scores → softmax → changing focus. After the plain-model walkthrough; no additional training is needed.
- **Part B:** train small GRU encoder–decoders with and without attention on synthetic digit reversal, then inspect free-running predictions and measured alignment.
- **Part C:** scaling, masks, greedy/beam search, and further practice.

CPU is sufficient; GPU is optional. No data downloads or API keys are required.
All 23 code cells passed a full default local CPU run (PyTorch 2.12.1+cpu), and the notebook schema was validated. This is not a Colab/GPU verification. The introduction's heatmap is explicitly hand-made; the trained model's heatmap is measured. Saved outputs are cleared for students.
