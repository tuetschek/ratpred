#
# Experiment management for Rating predictor
#

# Environment settings
SHELL = /bin/bash
RATPRED = ../run_ratpred.py
MEM = 12g
CPUS = 4

# XXX Replace this with your batch queue engine system command !
QP = --engine=sge --queue=cpu-troja.q
ifdef LOCAL
  QP = --engine=console
endif
QSUBMIT = qsubmit --logdir '$(TRY_DIR)' --mem $(MEM) --cores $(CPUS) --jobname R.$(TRY_NUM)$(CV_NUM)$(SEED).$(RUN_NAME) $(QP) $(QHOLD) --
# XXX

TRAIN_PORTION=1.0
ACTIVATE = export PYTHONPATH=""; source $$HOME/work/tgen_master/bin/activate # XXX source your activate script instead


ifdef DEBUG
  DEBUG_LOG_TRAIN = -d $(TRY_DIR)/train.debug-log.txt.gz
  DEBUG_LOG_TEST = -d $(TRY_DIR)/test.debug-log.txt
endif
ifdef DEBUG_TEST
  DEBUG_LOG_TEST = -d $(TRY_DIR)/test.debug-log.txt
endif
OVERALL_SCORES = ../util/overall_scores.py
GREP_SCORES = ../util/grep_scores.pl --dist-range=2
DESC_EXP = ../util/describe_experiment.py
CFG_OVERRIDE_TOOL = ../util/cfg_override.py
NOWRAP = ../util/nowrap.pl
NUM_EXP = 20  # number of experiments to list in `make desc'

LOGFILES = $$file/R.*test* $$file/{ratpred_test.,test.,}log.txt # logfile pattern to use for score printing
SCOREFILE = SCORE

# Runs directories
RUNS_DIR  := runs# main directory for experiment outputs
TRY_NUM   := $(shell perl -e '$$m=0; for(<$(RUNS_DIR)/*>){/\/(\d+)_/ and $$1 > $$m and $$m=$$1;} printf "%03d", $$m+1;')# experiment number
RUN_NAME  := experiment# default name, to be overridden by targets
DATE      := $(shell date +%Y-%m-%d_%H-%M-%S)
TRY_DIR    = $(RUNS_DIR)/$(TRY_NUM)_$(DATE)_$(RUN_NAME)# experiment main output directory (disregarding random seeds and CV)
TRAINING_SET := toy

ifdef D # Shortcut D -> DESC
  DESC := $(D)
endif
ifdef N # Shortcut N -> NUM_EXP
  NUM_EXP := $(N)
endif
ifdef R # Shortcut R -> REFRESH
  REFRESH := $(R)
endif

# Input data file defaults

CONFIG = config/config.yaml

TRAIN_DATA = data/$(TRAINING_SET)/$(CV_NUM)/train.tsv
VALID_DATA = data/$(TRAINING_SET)/$(CV_NUM)/devel.tsv
TEST_DATA = data/$(TRAINING_SET)/$(CV_NUM)/test.tsv
ifdef EVAL_TEST
    EVAL_IN = test.tsv
    EVAL_OUT = predicted-test.tsv
else
    EVAL_IN = devel.tsv
    EVAL_OUT = predicted-devel.tsv
endif

# if the training data set has "cv" in its name, default to looking for CV runs
ifneq (,$(findstring cv,$(TRAINING_SET)))
  CV_NUMS=($(shell find data/$(TRAINING_SET)/* -type d -printf " %f "))
else
  CV_NUMS=("")
endif

# Help text

define HELP_MESSAGE
Rating prediction experiments
================

Note that running experiments using this Makefile requires a batch queuing system,
such as Sun Grid Engine. You need to adjust your queuing engine command in
this Makefile code before using the commands.

Experiments are stored in the "runs/" subdirectory and given consecutive
numbers.

make desc [NUM_EXP=20] [REFRESH=1]
* List experiments with descriptions and (possibly) scores.
* Change NUM_EXP if you need to see more than 20 latest results.
* Refresh scores to obtain new results by setting REFRESH=1 (makes it slower).

make cv_run D='Description' TRAINING_SET=toy
* Run a full experiment where RatPred training is directly followed by
    testing.
  - set TRAINING_SET to the required subdirectory path in "data/"
    (if the directory path contains "cv", a cross-validation experiment
    is launched automatically)
  - set RANDS=1 to use 5 different random seeds (and average the results)
  - set EVAL_TEST=1 to evaluate on test data instead of development data

make train D='Description' TRAINING_SET=toy [EVAL_TEST=1]
* Same as the whole experiment, but training only (no testing)
  - if EVAL_TEST is set, "devel.tsv" is used for validation (instead of
    cutting out part of the training data)

make test TRAINING_SET=toy [EVAL_TEST=1]
* Testing only (despite the name, TRAINING_SET controls the test set
    subdirectory path in "data/")
    - "devel.tsv" is used for testing by default, "test.tsv" if EVAL_TEST is set

make score-XXX [EVAL_TEST=1]
* Retrieve a score for run without RANDS/cross-validation.
  - gets score on the test set if EVAL_TEST is set, development set otherwise

make cv_score-XXX [EVAL_TEST=1]
* Retrieve a score for run with RANDS or cross-validation.
  - gets score on the test set if EVAL_TEST is set, development set otherwise
---

Use ACTIVATE='source path/to/virtualenv/bin/activate' to run
within a Virtualenv instead of the default Python.

Use DEBUG=1 to activate debugging outputs. Use DEBUG_TEST=1 to activate
debugging only for testing (in CV runs).
endef
export HELP_MESSAGE

#
# Targets
#

# Auxiliary targets

help:
	@echo "$$HELP_MESSAGE" | egrep --color '^(\s*make.*|)'

# List all experiments, find scores in logs (of various versions)
desc:
	@COLS=`tput cols`; \
	ls -d $(RUNS_DIR)/* | sort | tail -n $(NUM_EXP) | while read file; do \
		if [[ -f $$file/$(SCOREFILE) && -z "$(REFRESH)" ]]; then \
			cat $$file/$(SCOREFILE) | $(NOWRAP) $$COLS; \
			continue ; \
		fi ; \
		echo -ne $$file | sed 's/runs\///;s/_/  /;s/_/ /;s/_.*/: /;s/  20/  /;s/-[0-9][0-9]  /  /;' > $$file/$(SCOREFILE);  \
		$(GREP_SCORES) $(LOGFILES) >> $$file/$(SCOREFILE) ; \
		cat $$file/ABOUT | tr  '\n' ' ' >> $$file/$(SCOREFILE); \
		echo >> $$file/$(SCOREFILE); \
		cat $$file/$(SCOREFILE) | $(NOWRAP) $$COLS; \
	done

desc_wrap:
	@ls -d $(RUNS_DIR)/* | sort | tail -n $(NUM_EXP) | while read file; do \
		cat $$file/$(SCOREFILE) ; \
	done


printvars:
	$(foreach V, $(sort $(.VARIABLES)), $(if $(filter-out environment% default automatic, $(origin $V)), $(info $V=$($V) ($(value $V)))))

printgit:
	@git status
	@echo -e "\n*** *** ***\n"
	@git log --pretty=format:"%h - %an, %ar : %s" -1
	@echo -e "\n*** *** ***\n"
	@git diff


prepare_dir:
	# create the experiment directory, print description and git version
	echo "PREPARE DIR" $(TRY_DIR) $(CV_NUM) $(SEED)
	mkdir -p $(TRY_DIR)
	cp $(CONFIG) $(TRY_DIR)/config.yaml
	if [[ -n '$(CFG_OVERRIDE)' ]]; then \
		$(CFG_OVERRIDE_TOOL) $(TRY_DIR)/config.yaml '$(CFG_OVERRIDE)' ; \
	fi
	CONFIG=$(TRY_DIR)/config.yaml ; \
	[[ -n "$(DEBUG)" || -n "$(DEBUG_TEST)" ]] && DEBUG_SETTING=-d ; \
	[[ -n "$(RANDS)" ]] && RANDS_SETTING=-r ; \
	$(DESC_EXP) -t "$(TRAINING_SET)" -p "$(TRAIN_PORTION)" -c "$(CV_NUMS)" $$DEBUG_SETTING $$RANDS_SETTING $$CONFIG > $(TRY_DIR)/ABOUT ; \
	echo "$(DESC)" >> $(TRY_DIR)/ABOUT
	make printvars > $(TRY_DIR)/VARS
	make printgit > $(TRY_DIR)/GIT_VERSION

prepare_data: prepare_dir
prepare_data:
	# copy train-dev-test data
	cp $(TRAIN_DATA) $(TRY_DIR)/train.tsv
	cp $(VALID_DATA) $(TRY_DIR)/devel.tsv
	cp $(TEST_DATA) $(TRY_DIR)/test.tsv


# Main targets

cv_run: RUN_NAME := cv_run
cv_run: prepare_dir
cv_run:
	CV_NUMS=$(CV_NUMS); \
	for CV_NUM in "$${CV_NUMS[@]}"; do \
		if [[ -n "$(RANDS)" ]]; then \
			SEEDS=(s0 s1 s2 s3 s4); \
		else \
			SEEDS=(""); \
		fi; \
		for SEED in "$${SEEDS[@]}"; do \
			TRAIN_RUN=train ; \
			TEST_RUN=test ; \
			make $$TRAIN_RUN TRY_DIR=$(TRY_DIR)/$$CV_NUM$$SEED SEED=$$SEED TRY_NUM=$(TRY_NUM) CV_NUM=$$CV_NUM CFG_OVERRIDE='$(CFG_OVERRIDE)' 2>&1 | tee -a $(TRY_DIR)/cv.log.txt ; \
			JOB_NUM=`cat $(TRY_DIR)/cv.log.txt | grep '^\(Your job\|Submitted batch\|Job_ID\)' | tail -n 1 | sed 's/^\(Your job\|Submitted batch job\|Job_ID:\) \([0-9]\+\).*/\2/;' ` ; \
			if [ -n "$$JOB_NUM" ]; then \
			    QHOLD="--hold $$JOB_NUM"; \
			fi ; \
			make $$TEST_RUN TRY_DIR=$(TRY_DIR)/$$CV_NUM$$SEED SEED=$$SEED TRY_NUM=$(TRY_NUM) CV_NUM=$$CV_NUM QP="$(QP)" QHOLD="$$QHOLD"; \
		done; \
	done

cv_score-%:
	$(eval TRY_DIR := $(shell ls -d $(RUNS_DIR)/$**))
	$(eval TRY_NUM := $*)
	make cv_score TRY_DIR=$(TRY_DIR) TRY_NUM=$(TRY_NUM) EVAL_TEST=$(EVAL_TEST)

cv_score:
	$(OVERALL_SCORES) -s $(TRY_DIR) $(EVAL_OUT) 2>&1 | tee $(TRY_DIR)/test.log.txt

score-%:
	$(eval TRY_DIR := $(shell ls -d $(RUNS_DIR)/$**))
	$(eval TRY_NUM := $*)
	make score TRY_DIR=$(TRY_DIR) TRY_NUM=$(TRY_NUM) EVAL_TEST=$(EVAL_TEST)

score:
	$(OVERALL_SCORES) $(TRY_DIR)/$(EVAL_OUT) 2>&1 | tee $(TRY_DIR)/test.log.txt

train: RUN_NAME := train
train:
	make prepare_data TRY_DIR=$(TRY_DIR) EVAL_TEST=$(EVAL_TEST) SEED=$(SEED) CV_NUM=$(CV_NUM) CFG_OVERRIDE='$(CFG_OVERRIDE)' 
	make train_process TRY_DIR=$(TRY_DIR) TRY_NUM=$(TRY_NUM) EVAL_TEST=$(EVAL_TEST) SEED=$(SEED) CV_NUM=$(CV_NUM) \
	    TRAIN_PORTION=$(TRAIN_PORTION) DEBUG_LOG_TRAIN=$(DEBUG_LOG_TRAIN)

train_process: RUN_NAME := train
train_process:
	if [[ -n "$(EVAL_TEST)" ]]; then \
		VALID_DATA_SPEC="-v \"$(TRY_DIR)/devel.tsv\""; \
	fi; \
	$(QSUBMIT) '$(ACTIVATE); \
		$(RATPRED) $(DEBUG_LOG_TRAIN) train \
		-p $(TRAIN_PORTION) -r "$(SEED)" '"$$VALID_DATA_SPEC"' \
		$(TRY_DIR)/config.yaml $(TRY_DIR)/train.tsv \
		$(TRY_DIR)/model.pickle.gz' 2>&1 | tee $(TRY_DIR)/train.log.txt

test: RUN_NAME := test
test: MODEL := $(TRY_DIR)/model.pickle.gz
test:
	# run
	$(QSUBMIT) '$(ACTIVATE); \
		$(RATPRED) $(DEBUG_LOG_TEST) test \
		-w $(TRY_DIR)/$(EVAL_OUT) $(MODEL) $(TRY_DIR)/$(EVAL_IN)' \
		2>&1 | tee $(TRY_DIR)/$(RUN_NAME).log.txt

# Rerun an experiment (TRY_DIR, TRY_NUM, RUN_NAME must be set!)
rerun:
	LAST_LOGFILE=`ls -t $(TRY_DIR)/R.*.$(RUN_NAME).* | head -n 1` ; \
	COMMAND=`cat $$LAST_LOGFILE | sed '1,/^== Directory/d;/^== Hard res/,$$d;s/^== Command:\s*//;'` ; \
	echo "Rerunning command: $$COMMAND"; \
	$(QSUBMIT) "$$COMMAND" 2>&1 | tee -a $(TRY_DIR)/$(RUN_NAME).log.txt ;

