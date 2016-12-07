#
# Experiment management for Rating predictor
#

# Environment settings
SHELL = /bin/bash
RATPRED = ../run_ratpred.py
MEM = 8g
CPUS = 4
QP = --queue longq
QSUBMIT = qsubmit --logdir '$(TRY_DIR)' --mem $(MEM) --cores $(CPUS) --jobname R.$(TRY_NUM).$(RUN_NAME) $(QP) $(QHOLD)# using OB's qsubmit script
ifdef DEBUG
  DEBUG_LOG_TRAIN = -d $(TRY_DIR)/train.debug-log.txt.gz
  DEBUG_LOG_TEST = -d $(TRY_DIR)/test.debug-log.txt
endif
ifdef DEBUG_TEST
  DEBUG_LOG_TEST = -d $(TRY_DIR)/test.debug-log.txt
endif
AVERAGE_SCORES = ../util/average_scores.pl
GREP_SCORES = ../util/grep_scores.pl --bleu-range=30:50 --nist-range=4:5
DESC_EXP = ../util/describe_experiment.pl
NUM_EXP = 20  # number of experiments to list in `make desc'

LOGFILES = $$file/R.* $$file/{ratpred_test.,}log.txt # logfile pattern to use for score printing
SCOREFILE = SCORE

# Runs directories
RUNS_DIR  := runs# main directory for experiment outputs
TRY_NUM   := $(shell perl -e '$$m=0; for(<$(RUNS_DIR)/*>){/\/(\d+)_/ and $$1 > $$m and $$m=$$1;} printf "%03d", $$m+1;')# experiment number
RUN_NAME  := experiment# default name, to be overridden by targets
DATE      := $(shell date +%Y-%m-%d_%H-%M-%S)
TRY_DIR    = $(RUNS_DIR)/$(TRY_NUM)_$(DATE)_$(RUN_NAME)# experiment output directory
TRAINING_SET := toy

ifdef D # Shortcut D -> DESC
  DESC := $(D)
endif

# Input data file defaults

CONFIG = config/config.py

TRAIN_DATA = data/$(TRAINING_SET)/train.tsv
TEST_DATA = data/$(TRAINING_SET)/devel.tsv

ifdef EVAL_TEST
  TEST_DATA = data/$(TRAINING_SET)/test.tsv
  VALID_DATA = data/$(TRAINING_SET)/devel.tsv
endif

# Help text

define HELP_MESSAGE
Rating prediction experiments
================

make desc [NUM_EXP=20] [REFRESH=1]
* List experiments with descriptions and (possibly) scores.
* Change NUM_EXP if you need to see more than 20 latest results.
* Refresh scores to obtain new results by setting REFRESH=1 (makes it slower).

make cv_run J=[XX] D='Description'
* Run an experiment where seq2seq
  training is directly followed by testing generation.
  - set RANDS=1 to use 5 different random seeds (and average the results)
  - set EVAL_TEST=1 to evaluate on test data instead of development data

make score-XXX
* Retrieve a score for run without RANDS.

make cv_score-XXX
* Retrieve a score for run with RANDS.
---

Use QP='--console' if you want the job to run interactively instead
of submitting to the cluster.

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
	@ls -d $(RUNS_DIR)/* | sort | tail -n $(NUM_EXP) | while read file; do \
		if [[ -f $$file/$(SCOREFILE) && -z "$(REFRESH)" ]]; then \
			cat $$file/$(SCOREFILE) ; \
			continue ; \
		fi ; \
		echo -ne $$file ":\t" | sed 's/runs\///;s/_/\t/;s/_/ /;s/_/\t/' > $$file/$(SCOREFILE);  \
		$(GREP_SCORES) $(LOGFILES) >> $$file/$(SCOREFILE) ; \
		echo -ne '    ' >> $$file/$(SCOREFILE) ; \
		cat $$file/ABOUT | tr  '\n' ' ' >> $$file/$(SCOREFILE); \
		echo >> $$file/$(SCOREFILE); \
		cat $$file/$(SCOREFILE); \
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
	mkdir -p $(TRY_DIR)
	CONFIG=$(CONFIG) ; \
	[[ -n "$(DEBUG)" || -n "$(DEBUG_TEST)" ]] && DEBUG_SETTING=-d ; \
	[[ -n "$(RANDS)" ]] && RANDS_SETTING=-r ; \
	[[ -n "$(EVAL_TEST)" ]] && EVAL_SETTING=-e ; \
	$(DESC_EXP) -t "$(TRAINING_SET)" -p "$(TRAIN_PORTION)" -c "$(CV_NUMS)" $$EVAL_SETTING $$DEBUG_SETTING $$RANDS_SETTING $$CONFIG > $(TRY_DIR)/ABOUT ; \
	echo "$(DESC)" >> $(TRY_DIR)/ABOUT
	make printvars > $(TRY_DIR)/VARS
	make printgit > $(TRY_DIR)/GIT_VERSION

# Main targets

# TODO do actual CV here
cv_run: RUN_NAME := cv_run
cv_run: prepare_dir
cv_run:
	if [[ -n "$(RANDS)" ]]; then \
		SEEDS=(s0 s1 s2 s3 s4); \
	else \
		SEEDS=(""); \
	fi; \
	for SEED in "$${SEEDS[@]}"; do \
		TRAIN_RUN=train ; \
		TEST_RUN=test_cv ; \
		make $$TRAIN_RUN TRY_DIR=$(TRY_DIR)/$$SEED SEED=$$SEED TRY_NUM=$(TRY_NUM) 2>&1 | tee -a $(TRY_DIR)/cv.log.txt ; \
		JOB_NUM=`cat $(TRY_DIR)/cv.log.txt | grep '^Your job' | tail -n 1 | sed 's/^Your job \([0-9]\+\).*/\1/;' ` ; \
		make $$TEST_RUN TRY_DIR=$(TRY_DIR)/$$SEED TRY_NUM=$(TRY_NUM) QP="$(QP)" QHOLD="--hold $$JOB_NUM"; \
	done

cv_score-%:
	$(eval TRY_DIR := $(shell ls -d $(RUNS_DIR)/$**))
	$(eval TRY_NUM := $*)
	make cv_score TRY_DIR=$(TRY_DIR) TRY_NUM=$(TRY_NUM)

cv_score:
	$(AVERAGE_SCORES) $(TRY_DIR)/*/R.*.*_test.o* | tee $(TRY_DIR)/test.log.txt

score-%:
	$(eval TRY_DIR := $(shell ls -d $(RUNS_DIR)/$**))
	$(eval TRY_NUM := $*)
	make score TRY_DIR=$(TRY_DIR) TRY_NUM=$(TRY_NUM)

score:
	$(AVERAGE_SCORES) $(TRY_DIR)/R.*.*_test.o* | tee $(TRY_DIR)/test.log.txt

train: RUN_NAME := train
train: prepare_dir
train:
	# copy needed files (to ensure replication), then run the experiment
	cp $(CONFIG) $(TRY_DIR)/config.py
	cp $(TRAIN_DATA) $(TRY_DIR)/train.tsv
	if [[ -n "$(EVAL_TEST)" ]]; then \
		cp $(VALID_DATA) $(TRY_DIR)/devel.tsv ; \
		VALID_DATA_SPEC="-v \"$(TRY_DIR)/devel.tsv\""; \
	fi; \
	$(QSUBMIT) '$(ACTIVATE); \
		$(RATPRED) $(DEBUG_LOG_TRAIN) train \
		-p $(TRAIN_PORTION) -r "$(SEED)" '"$$VALID_DATA_SPEC"' \
		$(TRY_DIR)/config.py $(TRY_DIR)/train.tsv \
		$(TRY_DIR)/model.pickle.gz' 2>&1 | tee $(TRY_DIR)/train.log.txt

test_cv:
	make prepare_test_data TRY_DIR=$(TRY_DIR)
	make test_process TRY_DIR=$(TRY_DIR) TRY_NUM=$(TRY_NUM) QP="$(QP)" QHOLD="$(QHOLD)"

prepare_test_data:
	cp $(TEST_DATA) $(TRY_DIR)/test.tsv

test_process: RUN_NAME := test
test_process: MODEL := $(TRY_DIR)/model.pickle.gz
test_process:
	# run
	$(QSUBMIT) '$(ACTIVATE); \
		$(RATPRED) $(DEBUG_LOG_TEST) test \
		-w $(TRY_DIR)/predicted.tsv $(MODEL) $(TRY_DIR)/test.tsv' \
		2>&1 | tee $(TRY_DIR)/$(RUN_NAME).log.txt

# Rerun an experiment (TRY_DIR, TRY_NUM, RUN_NAME must be set!)
rerun:
	LAST_LOGFILE=`ls -t $(TRY_DIR)/R.*.$(RUN_NAME).* | head -n 1` ; \
	COMMAND=`cat $$LAST_LOGFILE | sed '1,/^== Directory/d;/^== Hard res/,$$d;s/^== Command:\s*//;'` ; \
	echo "Rerunning command: $$COMMAND"; \
	$(QSUBMIT) "$$COMMAND" 2>&1 | tee -a $(TRY_DIR)/$(RUN_NAME).log.txt ;

