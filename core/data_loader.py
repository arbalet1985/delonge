from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd


@dataclass
class ColumnMap:
    rn: str | None
    x: str
    y: str
    ap: str | None
    ac: str | None


def read_excel_headers(file_path: str) -> list[str]:
    df = pd.read_excel(file_path, nrows=0)
    return [str(col) for col in df.columns]


def _normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def auto_detect_columns(headers: list[str]) -> ColumnMap | None:
    normalized = {_normalize_name(h): h for h in headers}

    x = normalized.get("x") or normalized.get("lon") or normalized.get("longitude")
    y = normalized.get("y") or normalized.get("lat") or normalized.get("latitude")
    ap = normalized.get("ap") or normalized.get("arp") or normalized.get("aр")
    ac = normalized.get("ac") or normalized.get("as") or normalized.get("aс")
    rn = normalized.get("rn") or normalized.get("rownum") or normalized.get("id")

    if not (x and y and (ap or ac)):
        return None
    return ColumnMap(rn=rn, x=x, y=y, ap=ap, ac=ac)


def load_points(file_path: str, column_map: ColumnMap) -> pd.DataFrame:
    columns = [column_map.x, column_map.y]
    if column_map.ap:
        columns.append(column_map.ap)
    if column_map.ac:
        columns.append(column_map.ac)
    if column_map.rn:
        columns.append(column_map.rn)

    df = pd.read_excel(file_path, usecols=columns)
    rename_map = {
        column_map.x: "x",
        column_map.y: "y",
    }
    if column_map.ap:
        rename_map[column_map.ap] = "ap"
    if column_map.ac:
        rename_map[column_map.ac] = "ac"
    if column_map.rn:
        rename_map[column_map.rn] = "rn"
    df = df.rename(columns=rename_map)

    numeric_cols = ["x", "y"]
    if "ap" in df.columns:
        numeric_cols.append("ap")
    if "ac" in df.columns:
        numeric_cols.append("ac")
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    required = ["x", "y"]
    if "ap" in df.columns:
        required.append("ap")
    if "ac" in df.columns:
        required.append("ac")
    df = df.dropna(subset=required).copy()
    if df.empty:
        raise ValueError("После очистки данных не осталось валидных строк.")

    if "rn" in df.columns:
        agg: dict[str, str] = {"rn": "first"}
        if "ap" in df.columns:
            agg["ap"] = "mean"
        if "ac" in df.columns:
            agg["ac"] = "mean"
        grouped = df.groupby(["x", "y"], as_index=False, sort=False).agg(agg)
    else:
        agg = {}
        if "ap" in df.columns:
            agg["ap"] = "mean"
        if "ac" in df.columns:
            agg["ac"] = "mean"
        grouped = df.groupby(["x", "y"], as_index=False, sort=False).agg(agg)
    if len(grouped) < 3:
        raise ValueError("Для триангуляции нужно минимум 3 уникальные точки.")
    return grouped.reset_index(drop=True)
