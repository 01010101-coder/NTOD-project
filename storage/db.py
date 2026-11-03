"""Открытие соединения с SQLite и явные границы транзакций.

Прагмы ``foreign_keys`` и ``journal_mode`` не персистентны между соединениями,
поэтому применяются в каждом :func:`connect` (решение design.md 2). Границы
транзакции задаёт наш код через :func:`transaction`, а не скрытое поведение
драйвера — для этого соединение открыто в режиме autocommit
(``isolation_level=None``).
"""

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager


def connect(path: str) -> sqlite3.Connection:
    """Открыть соединение с БД по ``path`` с прагмами проекта.

    Возвращает соединение с ``row_factory=sqlite3.Row`` (доступ к колонкам по
    именам) и ``isolation_level=None`` (autocommit: границы транзакций задаёт
    :func:`transaction`). Прагмы ``foreign_keys=ON`` и ``journal_mode=WAL``
    применяются после открытия (план M1 §6; WAL позволяет M4 читать во время
    записи).
    """
    conn = sqlite3.connect(path, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """Транзакционный блок: COMMIT при успехе, ROLLBACK при исключении.

    На входе исполняет ``BEGIN``, при нормальном выходе — ``COMMIT``, при
    исключении — ``ROLLBACK`` с пробросом исключения наружу (решение
    design.md 3). Вложенные транзакции не поддерживаются: вызов внутри
    незавершённой транзакции того же соединения приводит к ``ValueError``
    (SAVEPOINT в M1 не используются). Возвращает то же соединение.
    """
    if conn.in_transaction:
        raise ValueError("Вложенная транзакция не поддерживается (M1)")
    conn.execute("BEGIN")
    try:
        yield conn
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    conn.execute("COMMIT")
