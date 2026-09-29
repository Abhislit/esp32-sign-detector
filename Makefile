# ESP32-CAM sign-gesture detector
#
# make test        host tests: MHI C/Python parity + vote decoder
# make parity      just the MHI parity check
# make venv        create .venv with Python 3.12 (needed: TF has no 3.14 wheels)
# make check       device M0 verification (needs hardware)
# make collect     record a dataset from a device
# make inspect     sanity-check a recorded dataset
# make train       train + export model.h / labels.h / golden_vector.h
# make verify      verify the exported headers replay correctly
# make mock        run the mock device on :8099 for PC-side development
# make build       compile firmware
# make upload      flash firmware
# make clean

SHELL := /bin/bash
PY    ?= python3
VENV  := .venv
VPY   := $(VENV)/bin/python
URL   ?= http://192.168.4.1
DATA  ?= data/raw
REPS  ?= 100

.PHONY: test parity venv check collect inspect train verify mock build upload monitor clean distclean

test:
	./tools/run_tests.sh

parity:
	./tools/run_parity.sh

# TensorFlow has no wheels for Python 3.14, which is the system interpreter on
# some distros. uv fetches a 3.12 build if one is available.
venv:
	@if [ -x "$(VPY)" ]; then \
	    echo "$(VPY) already exists"; \
	else \
	    command -v uv >/dev/null || { echo "uv not found: https://astral.sh/uv"; exit 1; }; \
	    uv venv --python 3.12 $(VENV); \
	    uv pip install --python $(VPY) "tensorflow-cpu" numpy pillow opencv-python-headless; \
	    echo; echo "activate with:  source $(VENV)/bin/activate"; \
	fi

check:
	$(PY) pc/check_device.py --url $(URL)

collect:
	$(PY) pc/collect.py --url $(URL) --data $(DATA) --reps $(REPS)

inspect:
	$(PY) pc/inspect_data.py --data $(DATA)

train:
	$(VPY) pc/train.py --data $(DATA)

verify:
	$(VPY) tools/verify_export.py

mock:
	$(PY) tools/mock_device.py --port 8099

build:
	pio run

upload: build
	pio run -t upload

monitor:
	pio device monitor

clean:
	rm -rf build
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

distclean: clean
	rm -rf $(VENV) data/raw data/preview data/model_manifest.json
