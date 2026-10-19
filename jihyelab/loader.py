"""data.jihye.kr용 데이터셋 로더.

사용 예
-------
from jihyelab import upload_data
data = upload_data("boston")

서버의 metadata.json 예
----------------------
{
  "boston": {
    "desc": "보스턴 인근 지역 정보를 바탕으로 주택 가격을 예측하는 데이터",
    "url": "datasets/boston.csv",
    "index": null,
    "metadata": {
      "CRIM": "타운별 1인당 범죄율",
      "MEDV": "주택 가격 중앙값(단위: $1,000)"
    }
  }
}

여러 파일 데이터셋은 ``url`` 대신 다음처럼 ``files``를 사용한다.
"files": {"orders": "datasets/olist/orders.csv", "customers": "..."}
"""

from __future__ import annotations

import json
import shutil
import tempfile
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import unquote, urljoin, urlparse

import pandas as pd
import requests


BASE_URL = "https://data.jihye.kr/"
METADATA_FILE = "metadata.json"
SUPPORTED_EXTENSIONS = {
    ".csv", ".json", ".xls", ".xlsx", ".parquet", ".hossam", ".zip", ".shp"
}


def _is_url(value: str) -> bool:
    return urlparse(value).scheme.lower() in {"http", "https"}


def _metadata_location(local: str | Path | None) -> str | Path:
    if local is None:
        return urljoin(BASE_URL.rstrip("/") + "/", METADATA_FILE)
    path = Path(local).expanduser()
    return path if path.name.lower() == METADATA_FILE else path / METADATA_FILE


def _read_metadata(local: str | Path | None = None, timeout: float = 30) -> dict[str, Any]:
    location = _metadata_location(local)
    try:
        if isinstance(location, str) and _is_url(location):
            response = requests.get(location, timeout=timeout)
            response.raise_for_status()
            data = response.json()
        else:
            with Path(location).open("r", encoding="utf-8-sig") as file:
                data = json.load(file)
    except requests.RequestException as exc:
        raise ConnectionError(f"메타데이터를 가져오지 못했습니다: {location}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"metadata.json의 JSON 형식이 올바르지 않습니다: {location}") from exc
    except OSError as exc:
        raise FileNotFoundError(f"metadata.json을 읽을 수 없습니다: {location}") from exc

    if not isinstance(data, dict):
        raise ValueError("metadata.json의 최상위 값은 객체(dict)여야 합니다.")
    return data


def _resolve_path(value: str, local: str | Path | None) -> str:
    if _is_url(value):
        return value.replace("\\", "/")
    if local is None:
        return urljoin(BASE_URL.rstrip("/") + "/", value.replace("\\", "/"))
    base = Path(local).expanduser()
    if base.name.lower() == METADATA_FILE:
        base = base.parent
    return str((base / value).resolve())


def _download(url: str, timeout: float) -> bytes:
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        return response.content
    except requests.RequestException as exc:
        raise ConnectionError(f"데이터 파일을 내려받지 못했습니다: {url}") from exc


def _apply_index(data: pd.DataFrame, index_col: str | int | None) -> pd.DataFrame:
    if index_col is None:
        return data
    if isinstance(index_col, int):
        try:
            index_col = str(data.columns[index_col])
        except IndexError as exc:
            raise KeyError(f"인덱스 열 번호가 범위를 벗어났습니다: {index_col}") from exc
    if index_col not in data.columns:
        raise KeyError(f"인덱스로 지정한 열을 찾을 수 없습니다: {index_col}")
    return data.set_index(index_col)


def _read_excel(source: str | BytesIO, index_col: str | int | None) -> pd.DataFrame:
    book = pd.ExcelFile(source)
    data_sheets = [name for name in book.sheet_names if name.lower() != "metadata"]
    if not data_sheets:
        raise ValueError("Excel 파일에 데이터 시트가 없습니다.")
    return pd.read_excel(book, sheet_name=data_sheets[0], index_col=index_col)


def _safe_extract(archive: zipfile.ZipFile, destination: Path) -> None:
    root = destination.resolve()
    for member in archive.infolist():
        target = (destination / member.filename).resolve()
        if root != target and root not in target.parents:
            raise ValueError(f"안전하지 않은 ZIP 경로가 포함되어 있습니다: {member.filename}")
    archive.extractall(destination)


def _load_zip(path: str, index_col: str | int | None, timeout: float) -> pd.DataFrame | dict[str, pd.DataFrame]:
    temp_dir = Path(tempfile.mkdtemp(prefix="jihyelab_"))
    try:
        archive_path = temp_dir / "dataset.zip"
        if _is_url(path):
            archive_path.write_bytes(_download(path, timeout))
        else:
            archive_path = Path(path)
        extract_dir = temp_dir / "extracted"
        extract_dir.mkdir()
        with zipfile.ZipFile(archive_path) as archive:
            _safe_extract(archive, extract_dir)

        files = sorted(
            file for file in extract_dir.rglob("*")
            if file.is_file() and file.suffix.lower() in SUPPORTED_EXTENSIONS - {".zip"}
            and file.name.lower() != METADATA_FILE
            and not file.name.startswith(".")
        )
        if not files:
            raise FileNotFoundError("ZIP 파일 안에 지원되는 데이터 파일이 없습니다.")
        loaded = {file.stem: _load_file(str(file), index_col, timeout) for file in files}
        return next(iter(loaded.values())) if len(loaded) == 1 else loaded
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def _load_file(path: str, index_col: str | int | None = None, timeout: float = 30) -> pd.DataFrame | dict[str, pd.DataFrame]:
    suffix = Path(unquote(urlparse(path).path)).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"지원하지 않는 파일 형식입니다: {suffix or '(확장자 없음)'}")
    if suffix == ".zip":
        return _load_zip(path, index_col, timeout)

    remote = _is_url(path)
    content = _download(path, timeout) if remote and suffix != ".shp" else None

    if suffix == ".csv":
        return pd.read_csv(BytesIO(content) if content is not None else path, index_col=index_col)
    if suffix == ".json":
        data = pd.read_json(BytesIO(content) if content is not None else path)
        return _apply_index(data, index_col)
    if suffix in {".xls", ".xlsx"}:
        return _read_excel(BytesIO(content) if content is not None else path, index_col)
    if suffix in {".parquet", ".hossam"}:
        data = pd.read_parquet(BytesIO(content) if content is not None else path)
        return _apply_index(data, index_col)
    if suffix == ".shp":
        try:
            import geopandas as gpd
        except ImportError as exc:
            raise ImportError("SHP 파일을 읽으려면 geopandas를 설치해야 합니다.") from exc
        if remote:
            raise ValueError("원격 SHP는 관련 파일(.shx, .dbf 등)과 함께 ZIP으로 제공해 주세요.")
        return _apply_index(gpd.read_file(path), index_col)
    raise AssertionError("처리되지 않은 파일 형식")


def pretty_table(data: pd.DataFrame, tablefmt: str = "simple", headers: str | list[str] = "keys") -> None:
    """DataFrame을 간단한 표로 출력한다."""
    try:
        from tabulate import tabulate
        print(tabulate(data, headers=headers, tablefmt=tablefmt, showindex=True, numalign="right"))
    except ImportError:
        print(data.to_string(index=True))


def _print_dataset_info(info: Mapping[str, Any], view_url: bool, paths: Mapping[str, str]) -> None:
    desc = info.get("desc") or info.get("description")
    if desc:
        print(f"\033[94m📚 {desc}\033[0m")
    if view_url:
        for name, path in paths.items():
            label = f" ({name})" if len(paths) > 1 else ""
            print(f"\033[94m🌐 URL{label}: {path}\033[0m")

    fields = info.get("metadata") or info.get("fields")
    if fields:
        if isinstance(fields, Mapping):
            rows = list(fields.items())
        elif isinstance(fields, list):
            rows = [
                (item.get("field") or item.get("name"), item.get("description") or item.get("desc", ""))
                for item in fields if isinstance(item, Mapping)
            ]
        else:
            raise ValueError("metadata/fields 값은 객체 또는 목록이어야 합니다.")
        print()
        pretty_table(pd.DataFrame(rows, columns=["field", "description"]))
        print()


def load_info(search: str | None = None, local: str | Path | None = None, timeout: float = 30) -> pd.DataFrame:
    """사용 가능한 데이터셋의 이름, 설명, 주소를 DataFrame으로 반환한다."""
    metadata = _read_metadata(local, timeout)
    rows: list[dict[str, Any]] = []
    for name, raw_info in metadata.items():
        if not isinstance(raw_info, Mapping):
            raise ValueError(f"'{name}' 항목은 객체(dict)여야 합니다.")
        urls = raw_info.get("files", raw_info.get("url", ""))
        if isinstance(urls, Mapping):
            url_text = ", ".join(_resolve_path(str(v), local) for v in urls.values())
        else:
            url_text = _resolve_path(str(urls), local) if urls else ""
        rows.append({"name": name, "desc": raw_info.get("desc", raw_info.get("description", "")), "url": url_text})
    result = pd.DataFrame(rows, columns=["name", "desc", "url"])
    if search:
        result = result[result["name"].str.contains(search, case=False, regex=False, na=False)]
    return result.reset_index(drop=True)


def load_data(
    key: str,
    index_col: str | int | None = None,
    local: str | Path | None = None,
    *,
    view_url: bool = False,
    show_info: bool = True,
    timeout: float = 30,
) -> pd.DataFrame | dict[str, pd.DataFrame]:
    """파일 경로 또는 metadata.json의 키로 데이터셋을 불러온다."""
    direct_suffix = Path(unquote(urlparse(key).path)).suffix.lower()
    if direct_suffix in SUPPORTED_EXTENSIONS:
        return _load_file(key, index_col, timeout)

    metadata = _read_metadata(local, timeout)
    normalized = key.strip().casefold()
    matched_key = next((name for name in metadata if str(name).casefold() == normalized), None)
    if matched_key is None:
        available = ", ".join(map(str, metadata.keys()))
        raise KeyError(f"'{key}' 데이터셋을 찾을 수 없습니다. 사용 가능: {available}")
    info = metadata[matched_key]
    if not isinstance(info, Mapping):
        raise ValueError(f"'{matched_key}' 항목은 객체(dict)여야 합니다.")

    default_index = info.get("index")
    selected_index = index_col if index_col is not None else default_index
    raw_files = info.get("files")
    if raw_files is not None:
        if not isinstance(raw_files, Mapping) or not raw_files:
            raise ValueError(f"'{matched_key}'의 files는 비어 있지 않은 객체여야 합니다.")
        paths = {str(name): _resolve_path(str(value), local) for name, value in raw_files.items()}
        if show_info:
            _print_dataset_info(info, view_url, paths)
        result: dict[str, pd.DataFrame] = {}
        per_file_indexes = info.get("indexes", {})
        for name, path in paths.items():
            file_index = selected_index
            if index_col is None and isinstance(per_file_indexes, Mapping):
                file_index = per_file_indexes.get(name, default_index)
            loaded = _load_file(path, file_index, timeout)
            if isinstance(loaded, dict):
                for nested_name, frame in loaded.items():
                    result[f"{name}_{nested_name}"] = frame
            else:
                result[name] = loaded
        return result

    if not info.get("url"):
        raise ValueError(f"'{matched_key}' 항목에 url 또는 files가 없습니다.")
    path = _resolve_path(str(info["url"]), local)
    if show_info:
        _print_dataset_info(info, view_url, {str(matched_key): path})
    return _load_file(path, selected_index, timeout)


def upload_data(key: str, index_col: str | int | None = None, local: str | Path | None = None, **kwargs: Any) -> pd.DataFrame | dict[str, pd.DataFrame]:
    """사용자가 정한 호출 이름. 실제 동작은 load_data와 같다."""
    return load_data(key, index_col=index_col, local=local, **kwargs)


load_dataset = load_data


__all__ = [
    "BASE_URL", "load_data", "load_dataset", "load_info", "pretty_table", "upload_data"
]
