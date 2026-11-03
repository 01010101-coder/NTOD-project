# Makefile проекта «taxi-demand-forecast» (веха M1, change m1-tooling).

.PHONY: install test lint fmt

# При отсутствии .venv — создать, затем установить пакет с dev-зависимостями;
# повторный запуск в существующем окружении идемпотентен.
install:
	test -d .venv || python3.11 -m venv .venv
	.venv/bin/python -m pip install -e ".[dev]"

# Тесты (pytest собирает их из tests/).
test:
	.venv/bin/python -m pytest

# Линт: правила + проверка форматирования (падает при любом отклонении).
lint:
	.venv/bin/ruff check .
	.venv/bin/ruff format --check .

# Автоформатирование и безопасные исправления линта.
fmt:
	.venv/bin/ruff format .
	.venv/bin/ruff check --fix
