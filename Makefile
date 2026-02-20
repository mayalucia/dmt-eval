.PHONY: tangle test test-all clean help

CODEV_DIR := codev
ORG_FILES := $(wildcard $(CODEV_DIR)/*.org)

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

tangle: ## Tangle all codev/*.org files to produce src/dmt/**/*.py
	@for org in $(ORG_FILES); do \
		echo "Tangling $$org ..."; \
		emacs --batch -l org \
			--eval "(setq org-confirm-babel-evaluate nil)" \
			--eval "(org-babel-tangle-file \"$$org\")"; \
	done
	@echo "Done."

test: ## Run tests
	uv run --extra dev pytest test/ -v

clean: ## Remove build artifacts
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache/ dist/ *.egg-info/
