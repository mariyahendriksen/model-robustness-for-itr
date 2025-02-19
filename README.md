# Benchmark Granularity and Model Robustness for Image-Text Retrieval

This repository contains the code and experiment logs for the SIGIR submission **"Benchmark Granularity and Model Robustness for Image-Text Retrieval: A Reproducibility Study"**.


## Requirements

To set up the environment, install the requirements using the following command:

```angular2html
conda env create --file src/environment.yaml
```

This command will create a conda environment `evalvl`. Activate the created environment:

```angular2html
source activate evalvl
```

## Model evaluation

To evaluate a model, run this command:

```angular2html
python src/evaluation.py \
--dataset DATASET \
--model MODEL \
--task TASK \
--compute_from_scratch \
--perturbation PERTURBATION
```

### Explanation of Arguments:

  - --dataset (str): Specifies the dataset to use. (e.g., coco, f30k, etc.)
  - --model (str): Defines the model being evaluated. (e.g., align, clip)
  - --task (str): Sets the evaluation task. (e.g., `t2i` for text-to-image retrieval, `i2t` for image-to-text)
  - --compute_from_scratch (flag): If included, the evaluation will be performed from scratch instead of using cached results.
  - --perturbation (str): Applies a specific perturbation technique to test model robustness.

### Example Usage

```angular2html
python src/evaluation.py \
--dataset coco \
--model clip \
--task t2i \
--compute_from_scratch \
--perturbation char_swap
```


## Printing the Results

To print out the results, run the following command:

```angular2html
python src/results_printer.py
```


## Project Structure
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
