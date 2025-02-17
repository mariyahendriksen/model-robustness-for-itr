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
------------
├───config
│   ├───align
│   │   ├───coco
│   │   ├───coco_aug
│   │   ├───f30k
│   │   └───f30k_aug
│   ├───altclip
│   │   ├───coco
│   │   ├───coco_aug
│   │   ├───f30k
│   │   └───f30k_aug
│   ├───clip
│   │   ├───coco
│   │   ├───coco_aug
│   │   ├───f30k
│   │   └───f30k_aug
│   └───groupvit
│       ├───coco
│       ├───coco_aug
│       ├───f30k
│       └───f30k_aug
├───results
│   ├───coco
│   │   ├───align
│   │   │   ├───i2t
│   │   │   └───t2i
│   │   │       └───splits
│   │   ├───altclip
│   │   │   ├───i2t
│   │   │   └───t2i
│   │   ├───clip
│   │   │   ├───i2t
│   │   │   └───t2i
│   │   └───groupvit
│   │       ├───i2t
│   │       └───t2i
│   ├───coco_aug
│   │   ├───align
│   │   │   ├───i2t
│   │   │   └───t2i
│   │   ├───altclip
│   │   │   ├───i2t
│   │   │   └───t2i
│   │   ├───clip
│   │   │   ├───i2t
│   │   │   └───t2i
│   │   └───groupvit
│   │       ├───i2t
│   │       └───t2i
│   ├───f30k
│   │   ├───align
│   │   │   └───t2i
│   │   ├───altclip
│   │   │   └───t2i
│   │   ├───clip
│   │   │   ├───i2t
│   │   │   └───t2i
│   │   └───groupvit
│   │       └───t2i
│   └───f30k_aug
│       ├───align
│       │   ├───i2t
│       │   └───t2i
│       ├───altclip
│       │   ├───i2t
│       │   └───t2i
│       ├───clip
│       │   ├───i2t
│       │   └───t2i
│       └───groupvit
│           ├───i2t
│           └───t2i
└───src
    ├───data
    │   └───augmented
    ├───deprecated
    ├───evaluation
    ├───metrics
    ├───models
    │   ├───encoders
    │   └───relevance_estimators
    ├───perturbations
    │   └───perturbation_types
    ├───retrieval
    └───utils
```