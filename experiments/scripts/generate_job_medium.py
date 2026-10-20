"""Generate a medium-scale IMDb-like dataset (JOB-medium).

Produces tables with realistic statistical distributions modeled after
the IMDb dataset used in JOB benchmarks, but with synthetic data to
avoid copyright issues. Targets ~500-3000 rows per table.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path


def _zipf(n: int, alpha: float, rng: random.Random) -> int:
    """Sample from a discrete Zipf-like distribution over [1, n]."""
    weights = [1.0 / (i**alpha) for i in range(1, n + 1)]
    return rng.choices(range(1, n + 1), weights=weights, k=1)[0]


def generate(output_dir: Path, seed: int = 42) -> None:
    """Generate all JOB-medium tables."""
    rng = random.Random(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- title: 500 movies ---
    n_titles = 500
    kind_ids = [1] * 400 + [2] * 70 + [3] * 30  # movie, tv, video
    rng.shuffle(kind_ids)
    with (output_dir / "title.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["title_id", "production_year", "kind_id"])
        for i in range(1, n_titles + 1):
            year = rng.randint(1920, 2025)
            kind = kind_ids[i - 1] if i <= len(kind_ids) else 1
            w.writerow([i, year, kind])

    # --- person_info: 1500 persons ---
    n_persons = 1500
    genders = ["M", "F"]
    with (output_dir / "person_info.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["person_id", "gender", "birth_year"])
        for i in range(1, n_persons + 1):
            gender = rng.choice(genders)
            birth = rng.randint(1930, 2000)
            w.writerow([i, gender, birth])

    # --- cast_info: ~3000 rows, Zipf on title popularity ---
    n_cast = 3000
    with (output_dir / "cast_info.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cast_id", "person_id", "title_id", "role_id"])
        for i in range(1, n_cast + 1):
            person = rng.randint(1, n_persons)
            title = _zipf(n_titles, 1.1, rng)  # popular movies get more cast
            role = rng.randint(1, 4)
            w.writerow([i, person, title, role])

    # --- keyword: 200 keywords ---
    n_keywords = 200
    with (output_dir / "keyword.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["keyword_id"])
        for i in range(1, n_keywords + 1):
            w.writerow([i])

    # --- movie_keyword: ~2000 rows, Zipf on keywords ---
    n_mk = 2000
    with (output_dir / "movie_keyword.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mk_id", "title_id", "keyword_id"])
        for i in range(1, n_mk + 1):
            title = _zipf(n_titles, 0.8, rng)
            keyword = _zipf(n_keywords, 1.2, rng)
            w.writerow([i, title, keyword])

    # --- company_name: 100 companies ---
    n_companies = 100
    countries = ["US"] * 60 + ["UK"] * 15 + ["FR"] * 10 + ["DE"] * 8 + ["JP"] * 7
    with (output_dir / "company_name.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["company_id", "country_code"])
        for i in range(1, n_companies + 1):
            cc = countries[i - 1] if i <= len(countries) else "US"
            w.writerow([i, cc])

    # --- movie_companies: ~1000 rows ---
    n_mc = 1000
    with (output_dir / "movie_companies.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mc_id", "title_id", "company_id", "company_type_id"])
        for i in range(1, n_mc + 1):
            title = _zipf(n_titles, 0.9, rng)
            company = _zipf(n_companies, 1.3, rng)
            ctype = rng.randint(1, 2)
            w.writerow([i, title, company, ctype])

    # --- movie_info: ~1500 rows ---
    n_mi = 1500
    with (output_dir / "movie_info.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mi_id", "title_id", "info_type_id"])
        for i in range(1, n_mi + 1):
            title = _zipf(n_titles, 0.7, rng)
            info_type = rng.randint(1, 10)
            w.writerow([i, title, info_type])

    # Print summary
    total = (
        n_titles + n_persons + n_cast + n_keywords + n_mk + n_companies + n_mc + n_mi
    )
    print(f"Generated JOB-medium dataset in {output_dir}")
    print(f"  title:           {n_titles:>5d} rows")
    print(f"  person_info:     {n_persons:>5d} rows")
    print(f"  cast_info:       {n_cast:>5d} rows")
    print(f"  keyword:         {n_keywords:>5d} rows")
    print(f"  movie_keyword:   {n_mk:>5d} rows")
    print(f"  company_name:    {n_companies:>5d} rows")
    print(f"  movie_companies: {n_mc:>5d} rows")
    print(f"  movie_info:      {n_mi:>5d} rows")
    print(f"  TOTAL:           {total:>5d} rows")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/job_medium"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    generate(args.output_dir, args.seed)
