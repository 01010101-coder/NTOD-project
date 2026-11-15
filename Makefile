# Makefile проекта «taxi-demand-forecast» (веха M1, change m1-tooling).

.PHONY: install test lint fmt sim serve accept-m2 epoch epoch-check

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

# Генерация эталонной истории: 30 дней, seed 42, БД по умолчанию из конфига (§4 плана M2).
sim:
	.venv/bin/python -m simulator --days 30 --seed 42

# Сервер воспроизведения (change replay-server): флаги/дефолты — из секции app конфига.
serve:
	.venv/bin/python -m app

# Автоматическая приёмка вехи M2 (§11): скрипт — дословно приложение В плана M2;
# exit-код равен числу провалов по критериям A1–A13.
accept-m2:
	bash scripts/accept_m2.sh

# Эпоха данных M3 (change m3-dataset, решение 8 design.md): единый 37-дневный
# прогон мира в data/m3_epoch.db (первые train_days дней побитово = taxi.db,
# дни train_days+1.. — тест-продолжение). Число дней и путь БД — из секции ml
# конфига (подстановка через python -c, цель не рассинхронится с config.yaml);
# guard непустой trips в CLI даёт exit 2 при повторном запуске.
epoch:
	.venv/bin/python -m simulator --days $$(.venv/bin/python -c \
		"from common.config import load_config; m = load_config('config.yaml').ml; print(m.train_days + m.test_days)") \
		--db $$(.venv/bin/python -c "from common.config import load_config; print(load_config('config.yaml').ml.db)")

# Контроль целостности эпохи (DoD §12 плана M3): md5 CSV-выгрузок trips и
# zone_metrics_5min первых train_days дней data/m3_epoch.db против data/taxi.db;
# скрипт печатает совпадение по каждой таблице и падает при расхождении.
epoch-check:
	.venv/bin/python scripts/epoch_check.py
