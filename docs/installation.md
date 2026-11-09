# Installation and CLI use

## Local BIPIA detector

The window detector uses only the Python standard library. Download BIPIA
without the optional BrowseSafe-Bench files:

~~~powershell
.\scripts\download_datasets.ps1 -SkipBrowseSafe
py .\scripts\train_bipia_detector.py
~~~

Then scan retrieved content:

~~~powershell
py .\scripts\predict_bipia.py --task "Find the invoice total" --file .\retrieved.txt
~~~

You can pass inline text with --text instead of --file. The model and metrics
are saved under runs/bipia_detector.

## Qwen LoRA

Use Python 3.11 and a virtual environment. Install a PyTorch build suitable for
your OS and compute device using the official
[PyTorch installation selector](https://pytorch.org/get-started/locally/),
then install the project training dependencies:

~~~powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
~~~

Prepare data from official BIPIA train files and start LoRA fine-tuning:

~~~powershell
py .\scripts\prepare_bipia_sft.py
py .\scripts\train_qwen_sft.py --model Qwen/Qwen3-0.6B --method lora
~~~

The first run downloads the base model if it is not cached. Keep the prepared
official test split out of training.

## QLoRA and MELON embeddings

QLoRA requires a compatible CUDA PyTorch build and bitsandbytes. Linux or WSL2
is recommended:

~~~powershell
pip install -r requirements-qlora.txt
py .\scripts\train_qwen_sft.py --model Qwen/Qwen3-0.6B --method qlora
~~~

The paper's action comparator can use the optional OpenAI embeddings provider:
install requirements-melon-api.txt and set OPENAI_API_KEY in the process
environment. No key is stored in source or config files.
