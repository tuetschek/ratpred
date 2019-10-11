
RatPred Installation and Usage
==============================

Installation
------------

RatPred requires Python 2.7 and a [slightly outdated version](https://github.com/UFAL-DSG/tgen/tree/2977686cf32ce3dcf3c17c6acf11de1ccb6380b2) of [TGen](https://github.com/UFAL-DSG/tgen) to work (using [Tensorflow 1.13.1](https://pypi.python.org/pypi/tensorflow/1.13.1)). We recommend using a [virtualenv](https://virtualenv.pypa.io/en/stable/) for installation.

You can just clone RatPred and install the dependencies using `pip`:
```
git clone https://github.com/tuetschek/ratpred
pip install --process-dependency-links -r ratpred/requirements.txt
```

Usage
-----

### Training

To train RatPred on a given data file, run:

```
./run_ratpred.py train config.py train.csv model.pickle.gz
```
Here:
- `config.py` is a configuration file, which is loaded as Python code that creates
    a dictionary with all the settings. See 
    [experiments/config/config.py](experiments/config/config.py) for an example.

* `train.csv` is the input TSV file with training data, which uses the following columns:
    * `mr` -- training meaning representations (Cambridge dialogue acts as used 
        [in our datasets](https://github.com/jeknov/EMNLP_17_submission))
    * `system_ref` -- training NLG system outputs (tokenized texts)
    * `system_ref2` -- training NLG system outputs -- 2nd, worse reference (without loss of generality)
    * `quality` -- the training human quality ratings (depending on the `target_col` setting 
        in the configuration file, a different column may be used as the target ratings)
        * for ranking instances, a `1` in this column just indicates this criterion is used for training 
          -- the order is given just by `system_ref` vs. `system_ref2`
    * `is_real` -- (optional) 1/0 indicator if the given instance was synthesised
    * `orig_ref` -- (optional) human references for the given MR, separated by the special 
        `<|>` token

* `model.pickle.gz` is the target file where the trained model will be saved.

There are a few more optional parameters. Run `./run_ratpred.py train -h` for more information.

### Prediction

To test RatPred on a given file, run:

```
./run_ratpred.py test [-w predictions.tsv] model.pickle.gz test.tsv
```
Parameters:
* `model.pickle.gz` is the path to the model to be loaded.
* `test.tsv` is the TSV file to use for testing (same format as training file; 
    `is_real` and `orig_ref` columns are ignored)
* `-w predictions.tsv` is an (optional) file name where to save the predictions. If no output
    file is given, the program will only print overall scores on the console.


### Our Experiments

Preprocessing, configuration, and experiment management for our experiments used in the 
[ICML-LGNL 2017](https://arxiv.org/abs/1708.01759) and [INLG 2019](https://arxiv.org/abs/1910.04731) papers is stored in the [experiments/](experiments/) subdirectory.

#### NEM data ####

* We used our [EMNLP 2017](https://arxiv.org/abs/1707.06875) collection of rated NLG system outputs, which 
    can be downloaded from [Jekaterina's GitHub page](https://github.com/jeknov/EMNLP_17_submission).
    * use the [emnlp_data_individual_hum_scores.csv](https://github.com/jeknov/EMNLP_17_submission/raw/master/emnlp_data_individual_hum_scores.csv) file as the source for conversion (medians are computed internally during conversion).

* As additional data for synthesis, we used the original sources and converted them to TSVs
    with two columns -- `mr` (the dialogue act meaning representation) and `orig_ref` with
    the corresponding human reference. You can download the sets here:
    * [BAGEL](http://farm2.user.srcf.net/research/bagel/)
    * [SFHot + SFRest](https://www.repository.cam.ac.uk/handle/1810/251304)

* Preprocessing and data synthesis is controlled by the 
    [create_v3.py](experiments/input/create_v3.py) script. This is a wrapper around the main
    [data format conversion script](experiments/input/convert.py), calling it in all required
    configurations. 
    
    Please see the [create_v3.py](experiments/input/create_v3.py) code
    and adjust paths as necessary before running it (paths to the source dataset, additional
    data for synthesis, and converted data storage are hardcoded!)

#### E2E data ####

* The [E2E Challenge](https://arxiv.org/abs/1901.07931) quality human ranking was used for ranking experiments. The data can be downloaded [here from Github](https://github.com/tuetschek/e2e-eval).
   * Get the [human_ratings/quality.csv](https://github.com/tuetschek/e2e-eval/raw/master/human_ratings/quality.csv) file, this is the source for conversion in our experiments. Download it into a [experiments/input/nlg-datasets](experiments/input/nlg-datasets) directory.
   
* Preprocessing and data synthesis is controlled by the [create_v4.py](experiments/input/create_v4.py) script, which calls the [data conversion](experiments/input/convert_e2e.py) in all needed configurations. This script must be adjusted before running -- a lot of things are hardcoded!

#### Experiment management ####

* There is a basic experiment management system in the [Makefile](experiments/Makefile).
    Note that it contains our site-specific code and needs to be adjusted before use. Learn
    more by running `make help`.

