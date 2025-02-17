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


## Project Organization
````
├───config
├───results
└───src
    │   environment.yaml
    │   evaluation.py
    │   results_printer.py
    │   retriever.py
    │   split_printer.py
    │   __init__.py
    │
    ├───data
    │       dataset.py
    │       __init__.py
    │
    ├───evaluation
    │       evaluator.py
    │       __init__.py
    │
    ├───metrics
    │       dcg.py
    │       recall_at_k.py
    │
    ├───models
    │   │   __init__.py
    │   │
    │   ├───encoders
    │   │       align.py
    │   │       altclip.py
    │   │       clip.py
    │   │       groupvit.py
    │   │
    │   └───relevance_estimators
    │           clip_based.py
    │
    ├───perturbations
    │   │   perturbation.py
    │   │
    │   └───perturbation_types
    │           ARO.py
    │           distraction_based.py
    │           synonym_based.py
    │           typos.py
    │
    ├───retrieval
    │       retriever.py
    │       __init__.py
    │
    └───utils
            dataset_preprocessing.py
            image_processing.py
            multimodal.py
            utils.py
