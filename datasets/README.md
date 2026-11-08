# Dataset folders

Run `..\scripts\download_datasets.ps1` from the repository root. It creates this layout and downloads source files into the appropriate area:

```text
datasets/
  gmail/
    raw/       BIPIA EmailQA train/test and attack prompt files
    prepared/  normalized examples made from train only
  internet/
    raw/       BrowseSafe-Bench train/test Parquet files
    prepared/  normalized browser examples made from train only
```

Keep test sets held out for evaluation. `raw/` and `prepared/` contents are excluded from Git; this README and the scripts are committed so teammates can reproduce the downloads. Preserve each upstream license and cite the source.

Sources: [BIPIA](https://github.com/microsoft/BIPIA), [BrowseSafe-Bench](https://huggingface.co/datasets/perplexity-ai/browsesafe-bench).
