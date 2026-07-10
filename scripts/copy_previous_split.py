"""Copy the prior DREAMT project split assignments into this project."""

from __future__ import annotations

from src.data.splits import (
    copy_previous_project_splits,
)


def main() -> None:
    output = copy_previous_project_splits()
    print(f"Copied prior project split assignments to {output}")


if __name__ == "__main__":
    main()
