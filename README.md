# Do Image-Text Retrieval Benchmarks Reflect the Performance of Vision-Language Models?

This repository contains the experiment logs for the SIGIR submission "Benchmark Granularity and Model Robustness for Image-Text Retrieval: A Reproducibility Study".


## Requirements

To set up the environment, install the requirements using the following command:

```angular2html
conda env create --file src/environment.yaml
```

This command will create a conda environment `evalvl`. Activate the created environment:

```angular2html
source activate evalvl
```

## Printing the results

To print out the results, run the following command:

```angular2html
python src/results_printer.py
```
