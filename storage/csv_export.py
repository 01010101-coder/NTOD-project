"""Экспорт таблиц контракта в CSV (решение плана M1 §7.5 — экспорт живёт в storage).

stdlib ``csv.writer`` поверх ``SELECT *`` (решение design.md 7): pandas
отвергнут — он приводит NULL/float к своим представлениям (NaN, переформатирование),
а экспорт обязан побайтово повторять хранимые строки. NULL выгружается пустым
полем; первая строка файла — имена колонок.
"""

import csv
import sqlite3

from storage.schema import TABLES


def export_table(conn: sqlite3.Connection, table: str, out_path: str) -> None:
    """Выгрузить таблицу ``table`` в CSV-файл ``out_path``.

    Имя таблицы принимается только из белого списка ``storage.schema.TABLES``
    (иначе ``ValueError`` — решение design.md 6/7: SQL строится подстановкой
    лишь проверенного имени, что исключает инъекцию через имя таблицы).
    Первая строка файла — заголовок из ``cursor.description``, далее — те же
    строки, что хранятся в таблице; NULL-значения выгружаются пустыми полями.
    """
    if table not in TABLES:
        raise ValueError(
            f"Неизвестная таблица {table!r}; разрешены только таблицы контракта: "
            + ", ".join(TABLES)
        )
    cursor = conn.execute(f"SELECT * FROM {table}")
    header = [column[0] for column in cursor.description]
    with open(out_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        for row in cursor:
            writer.writerow(["" if value is None else value for value in row])
