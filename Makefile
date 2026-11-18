# Makefile проекта «taxi-demand-forecast» (веха M1, change m1-tooling).

.PHONY: install test lint fmt sim serve accept-m2 epoch epoch-check world

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

# Сервер воспроизведения (change replay-server): флаги/дефолты — из секции app
# конфига. CONFIG — тот же параметр, что у `world`: финальный мир запускается
# как `make serve CONFIG=config.final.yaml` (change world-final-config).
serve: CONFIG ?= config.yaml
serve:
	.venv/bin/python -m app --config $(CONFIG)

# Автоматическая приёмка вехи M2 (§11): скрипт — дословно приложение В плана M2;
# exit-код равен числу провалов по критериям A1–A13.
accept-m2:
	bash scripts/accept_m2.sh

# Эпоха данных M3 (change m3-dataset, решение 8 design.md): единый 37-дневный
# прогон мира в data/m3_epoch.db (первые train_days дней побитово = taxi.db,
# дни train_days+1.. — тест-продолжение). Число дней и путь БД — из секции ml
# конфига (подстановка через python -c, цель не рассинхронится с config.yaml);
# guard непустой trips в CLI даёт exit 2 при повторном запуске. CONFIG —
# параметр мира (дефолт эталонный config.yaml; `make world` передаёт свой).
epoch: CONFIG ?= config.yaml
epoch:
	.venv/bin/python -m simulator --config $(CONFIG) \
		--days $$(.venv/bin/python -c \
		"from common.config import load_config; m = load_config('$(CONFIG)').ml; print(m.train_days + m.test_days)") \
		--db $$(.venv/bin/python -c "from common.config import load_config; print(load_config('$(CONFIG)').ml.db)")

# Сборка мира целиком по одному конфигу (change world-final-config, спека
# build-tooling): эпоха (цель `epoch`) → обучение M3 MLP/LSTM → сети
# планирования MLP/LSTM → δ-калибровка → калибровка порога сигнала → сетка
# веток A/B/C/D → сводка plan-eval. Каждый шаг получает `--config $(CONFIG)`,
# остановка на первой ошибке — штатное поведение make. Дефолт — финальный мир;
# эталонный мир собирается как `make world CONFIG=config.yaml` (пути и каталоги
# артефактов тогда прежние). Генерация веток — последовательная, не параллельная
# (design: опыт e16, swap на 8 ГБ).
world: CONFIG ?= config.final.yaml
world:
	$(MAKE) epoch CONFIG=$(CONFIG)
	.venv/bin/python -m ml train --config $(CONFIG) --arch mlp
	.venv/bin/python -m ml train --config $(CONFIG) --arch lstm
	.venv/bin/python -m ml plan-train --config $(CONFIG) --model mlp
	.venv/bin/python -m ml plan-train --config $(CONFIG) --model lstm
	.venv/bin/python -m simulator.policy_calibration --config $(CONFIG)
	.venv/bin/python -m simulator.policy_calibration --config $(CONFIG) --signal
	.venv/bin/python -m simulator.branches --config $(CONFIG) --all \
		--days $$(.venv/bin/python -c \
		"from common.config import load_config; m = load_config('$(CONFIG)').ml; print(m.train_days + m.test_days)")
	.venv/bin/python -m ml plan-eval --config $(CONFIG)

# Контроль целостности эпохи (DoD §12 плана M3): md5 CSV-выгрузок trips и
# zone_metrics_5min первых train_days дней data/m3_epoch.db против data/taxi.db;
# скрипт печатает совпадение по каждой таблице и падает при расхождении.
epoch-check:
	.venv/bin/python scripts/epoch_check.py
