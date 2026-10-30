# llm_gguf

A lightweight Python project that loads a GGUF-format language model and runs inference using MLX on Apple Silicon hardware. The project is built around a TinyLlama GGUF checkpoint bundled in this repository and provides a simple CLI for text generation.

## What this repo does

This project allows you to:

- load a GGUF model file with metadata and tensor weights
- translate the GGUF tensor names into the MLX model layout used by the `LlamaModel`
- reconstruct a SentencePiece tokenizer from the GGUF tokenizer metadata
- generate text tokens from a prompt with streaming output

In practice, the project implements a minimal LLM inference pipeline for GGUF models, with the main logic in:

- `src/main.py` — command-line entry point and generation loop
- `src/models.py` — model loading, weight translation, tokenizer creation, and generation logic
- `src/utils.py` — SentencePiece tokenizer builder

## How it works

The flow is straightforward:

1. The CLI reads arguments such as the GGUF path, prompt, max tokens, temperature, and seed.
2. `models.load(gguf_file)` loads the GGUF file with MLX and extracts metadata and weights.
3. The code inspects GGUF metadata such as model dimensions, layer count, vocab size, and tokenizer information.
4. It maps GGUF weight names to the equivalent MLX module names (`attn_q`, `ffn_up`, `token_embd`, etc. -> the internal Llama blocks used in the model class).
5. It applies model quantization when the GGUF format indicates a quantized checkpoint such as Q4 or Q8.
6. A `GGUFTokenizer` is created from the GGUF tokenizer metadata using SentencePiece.
7. During generation, the model predicts tokens autoregressively and prints generated text incrementally.

This is a small, direct implementation of GGUF inference rather than a full general-purpose model serving framework. It is best suited for local experimentation with compatible GGUF checkpoints.

## Included model

The repository ships with a sample compact model:

- `src/model/tinyllama-1.1b-chat-v1.0.Q4_0.gguf`

This is a Q4_0 GGUF version of TinyLlama and is intended as a ready-to-run demo model for the project.

## Dependencies

The project depends on the following packages:

- `mlx>=0.8` — Apple MLX runtime used for tensor operations and model execution
- `numpy` — numerical array support
- `protobuf==3.20.2` — required by SentencePiece tooling
- `sentencepiece` — tokenizer construction and encoding/decoding support

Python requirements:

- Python >= 3.9
- Recommended: macOS with Apple Silicon (M1/M2/M3/M4) for MLX compatibility

The project metadata in `pyproject.toml` declares the package name but the actual runtime packages are listed in `requirements.txt`.

## Project structure

```text
llm_gguf/
├── README.md
├── pyproject.toml
├── requirements.txt
├── uv.lock
├── src/
│   ├── main.py
│   ├── models.py
│   ├── utils.py
│   └── model/
│       └── tinyllama-1.1b-chat-v1.0.Q4_0.gguf
└── .venv/
```

## Local setup

### Option 1: Using Python venv

From the repository root:

```bash
cd /path/to/llm_gguf
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Option 2: Using uv

If you have `uv` installed:

```bash
cd /path/to/llm_gguf
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

### Option 3: System Python

If you already have a working Python environment:

```bash
cd /path/to/llm_gguf
python -m pip install -r requirements.txt
```

## Running the model

After installing dependencies, run the CLI with the bundled TinyLlama model:

```bash
cd /path/to/llm_gguf
source .venv/bin/activate
python src/main.py \
  --gguf src/model/tinyllama-1.1b-chat-v1.0.Q4_0.gguf \
  --prompt "Write a short poem about machine learning." \
  --max-tokens 80 \
  --temp 0.7 \
  --seed 123
```

You can also use the short flags:

```bash
python src/main.py --gguf src/model/tinyllama-1.1b-chat-v1.0.Q4_0.gguf --prompt "Hello there" -m 50 --temp 0.0
```

## Important notes for local execution

- The model file is expected to exist at the path passed with `--gguf`.
- The project uses MLX, which is designed primarily for Apple Silicon/macOS environments.
- The included model is a GGUF checkpoint; if you want to use a different model, it must be compatible with the same architecture and metadata structure expected by the code.
- If you change the model, you may need to use a tokenizer and architecture that match the model family.
- The inference loop is intentionally simple and prints generated text as it is produced.

## Example use cases

This repo is useful for:

- experimenting with GGUF model inference locally
- testing MLX-based decoding on Apple Silicon hardware
- learning how GGUF tokenizer + weight metadata can be loaded and used in Python
- prototyping lightweight generation flows before building a larger inference stack

## Notes

This project is intentionally minimal and not a production-grade model serving system. It focuses on the core mechanics of:

- GGUF loading
- model reconstruction
- tokenizer loading
- transformer iteration
- simple text generation

If you want to run this in a local environment, the minimum setup is:

1. install Python 3.9+
2. create a virtual environment
3. install the packages from `requirements.txt`
4. ensure an appropriate GGUF file is available
5. execute `src/main.py` with `--gguf` pointing to that file

This is enough to run the repository as-is on supported Apple Silicon hardware.
