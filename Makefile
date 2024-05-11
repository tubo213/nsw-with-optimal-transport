.PHONY: format lint run_all

# ruffでフォーマットを行う
format:
	rye run ruff format

# ruffでlintとフォーマットを行う
lint:
	rye run ruff check --fix; rye run mypy . --config-file pyproject.toml

# bin配下のシェルスクリプトを全て実行する
run_all:
	@find bin -type f -name "*.sh" -exec chmod +x {} \; -exec {} \;