"""Apply the shot schema and seed the owner row. Idempotent.

    uv run python -m ops.bootstrap.db
"""

from __future__ import annotations

import sqlalchemy as sa

from shot_core.db import apply_schema, make_engine
from shot_core.settings import get_settings


def main() -> None:
    s = get_settings()
    eng = make_engine(s.database_url)
    apply_schema(eng)
    with eng.begin() as cx:
        cx.execute(sa.text(
            "INSERT INTO shot.owners (phone) VALUES (:p) ON CONFLICT (phone) DO NOTHING"),
            {"p": s.owner_phone_number})
        n = cx.execute(sa.text(
            "SELECT count(*) FROM information_schema.tables WHERE table_schema='shot'"
        )).scalar_one()
    print(f"  schema applied: {n} tables in `shot`; owner {s.owner_phone_number} seeded")


if __name__ == "__main__":
    main()
