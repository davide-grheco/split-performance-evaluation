#!/usr/bin/env python3
"""
Summarize Snakemake benchmark .tsv files for all rules in an experiment.

Auto-discovers every subdirectory under experiments/<name>/benchmarks/ and
aggregates runtime (s) and peak RSS (max_rss, MiB) across all completed jobs.

Usage:
    python scripts/analyze_benchmarks.py
    python scripts/analyze_benchmarks.py --rule splits champion
    python scripts/analyze_benchmarks.py --save results/benchmarks.csv
    EXPERIMENT_CONFIG=config/experiment.toml python scripts/analyze_benchmarks.py
"""

import argparse
import glob
import os
import re
import sys
import tomllib

import pandas as pd

# Rules that include a {model} wildcard between dataset and splitter.
_RULES_WITH_MODEL = {"predictions", "layer3"}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _load_config(path: str) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------

def _parse_filename(
    path: str,
    rule: str,
    datasets: frozenset[str],
    splitters: list[str],
) -> dict | None:
    """
    Parse a benchmark TSV filename into metadata fields.

    Patterns (basename without .tsv):
      no model : {dataset}-{splitter}-ratio{ratio}-rep{rep}-fold{fold}
      with model: {dataset}-{model}-{splitter}-ratio{ratio}-rep{rep}-fold{fold}

    Splitters are matched longest-first so names containing hyphens
    (e.g. "spxy-jaccard") are handled correctly.
    """
    name = os.path.basename(path).removesuffix(".tsv")
    tail = re.search(r"-ratio(?P<ratio>[^-]+)-rep(?P<rep>\d+)-fold(?P<fold>\d+)$", name)
    if not tail:
        return None

    ratio  = tail.group("ratio")
    rep    = int(tail.group("rep"))
    fold   = int(tail.group("fold"))
    prefix = name[: tail.start()]   # everything before -ratio…

    # Match splitter name (longest first to handle compound names).
    splitter = None
    for s in splitters:
        if prefix.endswith("-" + s):
            splitter = s
            prefix   = prefix[: -(len(s) + 1)]
            break
    if splitter is None:
        return None

    model   = None
    dataset = prefix
    if rule in _RULES_WITH_MODEL:
        # prefix is now "{dataset}-{model}"; model names don't contain hyphens
        parts = prefix.rsplit("-", 1)
        if len(parts) != 2:
            return None
        dataset, model = parts

    if dataset not in datasets:
        print(f"  warning: unrecognised dataset '{dataset}' in {os.path.basename(path)}", file=sys.stderr)

    row = {"rule": rule, "dataset": dataset, "splitter": splitter,
           "ratio": ratio, "rep": rep, "fold": fold}
    if model is not None:
        row["model"] = model
    return row


def _load_rule(
    bmark_root: str,
    rule: str,
    datasets: frozenset[str],
    splitters: list[str],
) -> pd.DataFrame:
    frames = []
    for path in glob.glob(os.path.join(bmark_root, rule, "*.tsv")):
        meta = _parse_filename(path, rule, datasets, splitters)
        if meta is None:
            print(f"  warning: could not parse {path}", file=sys.stderr)
            continue
        try:
            df = pd.read_csv(path, sep="\t")
        except Exception as exc:
            print(f"  warning: could not read {path}: {exc}", file=sys.stderr)
            continue
        for k, v in meta.items():
            df[k] = v
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

_AGG = {
    "n"        : ("s",       "size"),
    "s_mean"   : ("s",       "mean"),
    "s_p95"    : ("s",       lambda x: x.quantile(0.95)),
    "s_max"    : ("s",       "max"),
    "rss_mean" : ("max_rss", "mean"),
    "rss_p95"  : ("max_rss", lambda x: x.quantile(0.95)),
    "rss_max"  : ("max_rss", "max"),
}


def _summarize(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    return (
        df.groupby(by)
        .agg(**_AGG)
        .sort_values(["rss_p95", "s_p95"], ascending=False)
        .round(2)
    )


def _worst_jobs(df: pd.DataFrame, sort_by: str, n: int = 20) -> pd.DataFrame:
    id_cols = [c for c in ["rule", "dataset", "model", "splitter", "ratio", "rep", "fold"]
               if c in df.columns]
    return (
        df[id_cols + ["s", "max_rss"]]
        .sort_values(sort_by, ascending=False)
        .head(n)
        .round(2)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _section(title: str, df: pd.DataFrame) -> None:
    bar = "=" * max(60, len(title) + 4)
    print(f"\n{bar}\n  {title}\n{bar}")
    if df.empty:
        print("  (no data)")
    else:
        print(df.to_string())
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Summarize Snakemake benchmark files for an experiment.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument(
        "--config", default=os.environ.get("EXPERIMENT_CONFIG", "config/experiment.toml"),
        metavar="PATH", help="Experiment TOML config (default: $EXPERIMENT_CONFIG or config/experiment.toml)",
    )
    ap.add_argument(
        "--exp-dir", default=None, metavar="DIR",
        help="Override the experiments/<name>/ root directory",
    )
    ap.add_argument(
        "--rule", nargs="+", default=None, metavar="RULE",
        help="Limit analysis to specific rule(s) (default: all discovered)",
    )
    ap.add_argument(
        "--save", default=None, metavar="PATH",
        help="Save combined DataFrame to a CSV file",
    )
    ap.add_argument(
        "--worst", type=int, default=20, metavar="N",
        help="Number of worst jobs to show (default: 20)",
    )
    args = ap.parse_args()

    cfg       = _load_config(args.config)
    datasets  = frozenset(cfg["data"]["datasets"])
    splitters = sorted(cfg["splitting"]["methods"], key=len, reverse=True)
    exp_name  = cfg["experiment"]["name"]
    exp_root  = args.exp_dir or os.path.join(cfg["output"]["root"], exp_name)
    bmark_root = os.path.join(exp_root, "benchmarks")

    if not os.path.isdir(bmark_root):
        print(f"error: benchmarks directory not found: {bmark_root}", file=sys.stderr)
        sys.exit(1)

    available = sorted(
        d for d in os.listdir(bmark_root)
        if os.path.isdir(os.path.join(bmark_root, d))
    )
    rules = args.rule or available
    unknown = sorted(set(rules) - set(available))
    if unknown:
        print(f"warning: no data for rule(s): {unknown}", file=sys.stderr)
        rules = [r for r in rules if r in set(available)]

    all_frames: list[pd.DataFrame] = []
    for rule in rules:
        print(f"Loading {rule}…", file=sys.stderr, end=" ")
        df = _load_rule(bmark_root, rule, datasets, splitters)
        if df.empty:
            print("(no data)", file=sys.stderr)
            continue
        all_frames.append(df)
        print(f"{len(df):,} rows", file=sys.stderr)

    if not all_frames:
        print("No benchmark data found.", file=sys.stderr)
        sys.exit(1)

    combined = pd.concat(all_frames, ignore_index=True)

    if args.save:
        os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
        combined.to_csv(args.save, index=False)
        print(f"\nSaved {len(combined):,} rows → {args.save}", file=sys.stderr)

    # --- Summaries ---
    _section("BY RULE  (s = wall-clock seconds, rss = peak MiB)", _summarize(combined, ["rule"]))
    _section("BY RULE × SPLITTER", _summarize(combined, ["rule", "splitter"]))
    _section("BY RULE × DATASET",  _summarize(combined, ["rule", "dataset"]))

    if "model" in combined.columns:
        model_df = combined[combined["model"].notna()]
        if not model_df.empty:
            _section("BY RULE × MODEL (rules with model wildcard only)",
                     _summarize(model_df, ["rule", "model"]))

    _section(f"WORST {args.worst} JOBS BY peak RSS (max_rss)",
             _worst_jobs(combined, "max_rss", args.worst))
    _section(f"WORST {args.worst} JOBS BY wall-clock time (s)",
             _worst_jobs(combined, "s", args.worst))


if __name__ == "__main__":
    main()
