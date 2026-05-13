# Installation

qkerasV3 is published on PyPI as **`qkeras-v3`** and imported as **`qkeras`**.

## Supported backends

qkerasV3 currently supports **TensorFlow** via **Keras 3**.

Before importing `qkeras`, ensure the Keras backend is set:

```bash
export KERAS_BACKEND=tensorflow
```

You can also set it per-command:

```bash
KERAS_BACKEND=tensorflow python -c "import qkeras; print(qkeras.__version__)"
```

## Install from PyPI

```bash
pip install -U pip
pip install qkeras-v3
```

## Install from source (development)

```bash
git clone https://github.com/fastmachinelearning/qkerasV3.git
cd qkerasV3
pip install -e .
```

## Verify installation

```bash
KERAS_BACKEND=tensorflow python - <<'PY'
import qkeras
print("qkeras import OK")
PY
```
