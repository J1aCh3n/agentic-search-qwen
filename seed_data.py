from __future__ import annotations

from seed_data_large import ensure_seed_data, seed_all_large


def seed_all() -> None:
    seed_all_large(force=True)


if __name__ == "__main__":
    seed_all()
