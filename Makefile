.PHONY: test run clean help

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

test:  ## Run the test suite
	python -m unittest discover -s tests -v

run:  ## Run the demo scenario
	python -m battousai.main

run-debug:  ## Run with debug logging
	python -m battousai.main --debug

run-long:  ## Run for 200 ticks
	python -m battousai.main --ticks 200

clean:  ## Remove build artifacts
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
	rm -rf build/ dist/ *.egg-info .pytest_cache
