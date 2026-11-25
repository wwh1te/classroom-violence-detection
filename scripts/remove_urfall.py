"""Remove UR Fall records and references from a repository checkout.

Run from the repository root with:
    python scripts/remove_urfall.py --apply

Without --apply the script reports the planned changes only. Local video files
belonging to removed records are moved into a dated backup folder under
``../../work/backups`` (outside the repository) when --apply is used.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from io import StringIO
import json
import os
from pathlib import Path
import shutil
import tempfile

LABEL_MAP = {"normal": "non-violent", "violence": "violent"}
REMOVED_IDS = ("urfall", "ur_fall", "ur-fall")


def is_removed(value: object) -> bool:
    text = str(value or "").casefold().replace(" ", "")
    return any(token in text for token in REMOVED_IDS)


def is_removed_row(row: dict) -> bool:
    return is_removed(row.get("dataset")) or is_removed(row.get("sample_id")) or is_removed(row.get("raw_path"))


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def write_json(path: Path, value: object) -> None:
    atomic_write(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    atomic_write(path, "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows))


def atomic_write(path: Path, content: str) -> None:
    if path.is_file() and path.read_text(encoding="utf-8-sig") == content:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".cleanup-tmp")
    with temp.open("w", encoding="utf-8", newline="") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def safe_repo_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if root.resolve() not in path.parents:
        raise ValueError(f"Refusing path outside repository: {relative}")
    return path


def migrate(root: Path, apply: bool) -> list[str]:
    changes: list[str] = []
    catalog_path = root / "data/metadata/catalog.jsonl"
    catalog = read_jsonl(catalog_path)
    kept_catalog = []
    catalog_changed = False
    for row in catalog:
        if is_removed_row(row):
            changes.append(f"remove catalog record {row.get('sample_id', '<unknown>')}")
            catalog_changed = True
            continue
        source_label = LABEL_MAP.get(row.get("label"), row.get("source_label", ""))
        if row.get("source_label") != source_label:
            row["source_label"] = source_label
            catalog_changed = True
        kept_catalog.append(row)
    if apply and catalog_changed:
        write_jsonl(catalog_path, kept_catalog)

    splits_path = root / "data/metadata/splits.csv"
    split_rows: list[dict] = []
    split_fields: list[str] = []
    splits_changed = False
    if splits_path.exists():
        with splits_path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            split_fields = list(reader.fieldnames or [])
            for row in reader:
                if is_removed_row(row):
                    changes.append(f"remove split row {row.get('sample_id', '<unknown>')}")
                    splits_changed = True
                    continue
                source_label = LABEL_MAP.get(row.get("label"), row.get("source_label", ""))
                if row.get("source_label") != source_label:
                    row["source_label"] = source_label
                    splits_changed = True
                split_rows.append(row)
        if "source_label" not in split_fields:
            split_fields.append("source_label")
            splits_changed = True
        if apply and splits_changed:
            buffer = StringIO(newline="")
            writer = csv.DictWriter(buffer, fieldnames=split_fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(split_rows)
            atomic_write(splits_path, buffer.getvalue())

    # Remove source records from local manifests and create a backup of their media.
    local_dir = root / "data/metadata/local"
    backup_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_dir = Path(tempfile.gettempdir()) / "classroom-violence-fall-detection-backups"
    if apply:
        backup_dir.mkdir(parents=True, exist_ok=True)
    for relative in (
        "data/metadata/local/downloads.jsonl",
        "data/metadata/local/manifest.jsonl",
        "data/metadata/local/preparation_errors.jsonl",
        "data/metadata/local/raw_normalization.jsonl",
        "data/metadata/local/raw_normalization_errors.jsonl",
        "data/metadata/local/audit.json",
        "data/annotations/local/events.csv",
    ):
        source = root / relative
        if apply and source.is_file():
            destination = backup_dir / f"{backup_stamp}__metadata__{relative.replace('/', '__')}"
            shutil.copy2(source, destination)
    removed_local_rows: list[dict] = []
    kept_manifest: list[dict] = []
    for name in ("downloads.jsonl", "manifest.jsonl", "preparation_errors.jsonl",
                 "raw_normalization.jsonl", "raw_normalization_errors.jsonl"):
        path = local_dir / name
        if not path.exists():
            continue
        rows = read_jsonl(path)
        removed_local_rows.extend(row for row in rows if is_removed_row(row))
        kept = [row for row in rows if not is_removed_row(row)]
        if name == "manifest.jsonl":
            kept_manifest = kept
        if len(kept) != len(rows):
            changes.append(f"filter {len(rows) - len(kept)} row(s) from data/metadata/local/{name}")
            if apply:
                write_jsonl(path, kept)

    media_paths = sorted({row.get(key) for row in removed_local_rows for key in ("raw_path", "processed_path") if row.get(key)})
    for media_root in (root / "data/raw", root / "data/processed"):
        if media_root.exists():
            media_paths.extend(path.relative_to(root).as_posix() for path in media_root.rglob("*")
                               if path.is_file() and is_removed(path.name))
    media_paths = sorted(set(media_paths))
    for relative in media_paths:
        source = safe_repo_path(root, relative)
        if source.is_file():
            changes.append(f"back up local media {relative}")
            if apply:
                destination = backup_dir / f"{backup_stamp}__media__{relative.replace('/', '__')}"
                shutil.move(str(source), str(destination))
    annotations = local_dir / "../annotations/local/events.csv"
    annotations = annotations.resolve()
    if annotations.is_file():
        with annotations.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            fields = list(reader.fieldnames or [])
            rows = list(reader)
        kept = [row for row in rows if not is_removed_row(row)]
        if len(kept) != len(rows):
            changes.append(f"filter {len(rows) - len(kept)} row(s) from data/annotations/local/events.csv")
            if apply:
                buffer = StringIO(newline="")
                writer = csv.DictWriter(buffer, fieldnames=fields)
                writer.writeheader()
                writer.writerows(kept)
                atomic_write(annotations, buffer.getvalue())

    local_audit_path = local_dir / "audit.json"
    if local_audit_path.exists():
        local_audit = json.loads(local_audit_path.read_text(encoding="utf-8-sig"))
        candidates = [pair for pair in local_audit.get("near_duplicate_candidates", [])
                      if not is_removed(pair.get("left")) and not is_removed(pair.get("right"))]
        local_counts: dict[str, int] = {}
        for row in kept_manifest:
            key = f"{row.get('split', '')}/{row.get('label', '')}"
            if row.get("split"):
                local_counts[key] = local_counts.get(key, 0) + 1
        local_warnings = [item for item in local_audit.get("warnings", [])
                          if item.get("type") != "pending_human_label_review"
                          and not is_removed(json.dumps(item, ensure_ascii=False))]
        local_warnings.insert(0, {"type": "pending_human_label_review",
                                  "count": sum(row.get("label_status") != "reviewed" for row in kept_manifest)})
        local_audit.update(sample_count=len(kept_manifest), warnings=local_warnings,
                           near_duplicate_candidates=candidates, counts=dict(sorted(local_counts.items())),
                           ready_for_final_evaluation=False)
        changes.append("filter local audit metadata using the remaining manifest")
        if apply:
            write_json(local_audit_path, local_audit)

    sources_path = root / "data/metadata/sources.json"
    if sources_path.exists():
        sources = json.loads(sources_path.read_text(encoding="utf-8-sig"))
        entries = sources.get("sources", [])
        kept_entries = [entry for entry in entries if not is_removed(json.dumps(entry, ensure_ascii=False))]
        if len(kept_entries) != len(entries):
            changes.append(f"remove {len(entries) - len(kept_entries)} source entry/entries from sources.json")
        sources["sources"] = kept_entries
        if apply:
            write_json(sources_path, sources)

    # Replace stale reports with counts derived from the filtered catalog/split metadata.
    counts: dict[str, int] = {}
    for row in split_rows:
        key = f"{row.get('split', '')}/{row.get('label', '')}"
        counts[key] = counts.get(key, 0) + 1
    catalog_counts: dict[str, int] = {}
    for row in kept_catalog:
        key = f"{row.get('dataset', '')}/{row.get('source_label', LABEL_MAP.get(row.get('label'), row.get('label', '')))}"
        catalog_counts[key] = catalog_counts.get(key, 0) + 1
    components = {row.get("split_component_id") for row in split_rows if row.get("split_component_id")}
    groups = {row.get("group_id") for row in kept_catalog if row.get("group_id")}

    audit_path = root / "docs/reports/audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8-sig")) if audit_path.exists() else {}
    old_candidates = audit.get("near_duplicate_candidates", [])
    candidates = [pair for pair in old_candidates if not is_removed(pair.get("left")) and not is_removed(pair.get("right"))]
    pending = len(kept_catalog)
    warnings = [item for item in audit.get("warnings", [])
                if item.get("type") != "pending_human_label_review" and not is_removed(json.dumps(item, ensure_ascii=False))]
    warnings.insert(0, {"type": "pending_human_label_review", "count": pending})
    audit.update(sample_count=len(kept_catalog), warnings=warnings, near_duplicate_candidates=candidates,
                 counts=dict(sorted(counts.items())), ready_for_final_evaluation=False)
    changes.append("rebuild audit summary from the filtered catalog and splits")
    if apply:
        write_json(audit_path, audit)

    stats_path = root / "docs/reports/dataset_statistics.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8-sig")) if stats_path.exists() else {}
    stats.update(videos=len(kept_catalog), source_counts=dict(sorted(catalog_counts.items())),
                 recording_groups=len(groups), split_components=len(components), split_counts=dict(sorted(counts.items())),
                 near_duplicate_candidates=len(candidates), cross_split_near_duplicate_candidates=sum(bool(p.get("cross_split")) for p in candidates),
                 pending_clip_review=pending, ready_for_final_evaluation=False,
                 raw_bytes=None, processed_bytes=None, source_duration_seconds=None,
                 processed_duration_seconds=None, processed_frames=None,
                 metrics_note="Các tổng dung lượng, thời lượng, frame và checksum phải được tính lại sau khi tải và chuẩn hóa catalog AIRTLab đã lọc.")
    changes.append("rebuild dataset statistics from the filtered catalog and splits; invalidate stale media totals")
    if apply:
        write_json(stats_path, stats)

    report_lines = [
        "# Mô tả dữ liệu và kết quả chuẩn bị", "",
        "Catalog được giới hạn ở AIRTLab. Nhãn nguồn được giữ riêng với nhãn nội bộ: `non-violent` → `normal`, `violent` → `violence`.",
        "Dataset AIRTLab có hai lớp; không có ví dụ fall.", "",
        f"Catalog có **{len(kept_catalog)} video** trong **{len(groups)} nhóm sự kiện**.", "",
        "| Nhãn nguồn | Video |", "| --- | ---: |",
    ]
    source_counts: dict[str, int] = {}
    for row in kept_catalog:
        label = row.get("source_label", "unknown")
        source_counts[label] = source_counts.get(label, 0) + 1
    report_lines.extend(f"| {label} | {count} |" for label, count in sorted(source_counts.items()))
    report_lines += ["", "## Chia tập", "", "| Tập | normal | violence | Tổng |", "| --- | ---: | ---: | ---: |"]
    for split in ("train", "val", "test"):
        normal = counts.get(f"{split}/normal", 0)
        violence = counts.get(f"{split}/violence", 0)
        report_lines.append(f"| {split} | {normal} | {violence} | {normal + violence} |")
    report_lines += [
        "", "## Trạng thái và giới hạn", "",
        "Các tổng dung lượng, thời lượng và số frame chưa được tính lại cho catalog hiện tại; cần tải và chuẩn hóa lại video trước khi báo cáo các chỉ số đó.",
        "Nhãn cấp clip cần được nhóm kiểm duyệt. AIRTLab được diễn trong một phòng, không đại diện riêng cho lớp học.",
        "Dataset này không có ví dụ fall; cần nguồn được phê duyệt trước khi huấn luyện hoặc đánh giá lớp fall.",
        "",
    ]
    if apply:
        atomic_write(root / "docs/reports/dataset_report.md", "\n".join(report_lines))
    changes.append("regenerate dataset report from filtered source and split metadata")

    return changes


def clean_docs(root: Path, apply: bool) -> list[str]:
    files = {
        "docs/dataset_sources.md": """# Nguồn dữ liệu và điều khoản sử dụng\n\nNgày kiểm tra: 17/10/2026. Phạm vi: nghiên cứu và giáo dục. Điều khoản dữ liệu độc lập với giấy phép mã nguồn.\n\n| Nguồn chính thức | Nhãn gốc | Nhãn nội bộ dự án | Điều khoản đã kiểm tra |\n| --- | --- | --- | --- |\n| [AIRTLab](https://github.com/airtlab/A-Dataset-for-Automatic-Violence-Detection-in-Videos/tree/1f7747e104301ccaa82ef5a2f6804b51ced1c398) | `non-violent` | `normal` | Dataset Release Agreement cho phép nghiên cứu và giáo dục, yêu cầu trích dẫn. Không gán giấy phép MIT/CC cho video; chưa xác lập quyền thương mại hoặc tái phân phối. |\n| AIRTLab | `violent` | `violence` | Cùng điều khoản và nguồn như trên. |\n| [RWF-2000](https://github.com/mchengny/RWF2000-Video-Database-for-Violence-Detection) | Chưa dùng | — | Không tải trong phiên bản này; không thay bằng bản mirror không rõ quyền. |\n\nAIRTLab có 350 clip từ hai camera: 120 clip non-violent và 230 clip violent. Cùng nhãn và số clip ở hai camera là cùng sự kiện; metadata ghép chúng bằng `group_id`. Nhóm non-violent có các hành vi dễ gây false positive như ôm, vỗ tay, ăn mừng và cử chỉ mạnh. Dataset được diễn trong một phòng, không phải video lớp học.\n\n`data/metadata/catalog.jsonl` lưu URL, nhãn nội bộ, nhãn gốc, camera, `group_id` và điều khoản. AIRTLab được ghim theo commit `1f7747e104301ccaa82ef5a2f6804b51ced1c398`; pipeline đối chiếu kích thước và SHA-1 Git blob trước khi chấp nhận tải. Không đưa video vào gói mã nguồn công khai; khả năng tải công khai không tự cấp quyền tái phân phối.\n\n## Trích dẫn\n\nBianculli, M., Falcionelli, N., Sernani, P., Tomassini, S., Contardo, P., Lombardi, M., & Dragoni, A. F. (2020). A dataset for automatic violence detection in videos. *Data in Brief, 33*, 106587. https://doi.org/10.1016/j.dib.2020.106587\n""",
        "docs/data_review.md": """# Phiếu kiểm duyệt dữ liệu — issue #1\n\nPhạm vi hiện tại: AIRTLab, nhãn gốc `non-violent`/`violent`, ánh xạ nội bộ `normal`/`violence`.\n\n## Kiểm duyệt nhãn và chất lượng\n\nChưa ghi nhận kết quả kiểm duyệt thủ công cho phiên bản catalog hiện tại. Chỉ đánh dấu video đã xem sau khi mở đúng file và ghi nhận xét cụ thể. Không sử dụng kết quả kiểm duyệt của một nguồn dữ liệu đã bị loại để kết luận cho phiên bản hiện tại.\n\n| Video đã xem | Nhãn nguồn | Mở được/chất lượng | Nhận xét nhãn | Quyết định |\n| --- | --- | --- | --- | --- |\n| Chưa kiểm tra | — | — | — | Chờ kiểm duyệt |\n\n## Kết luận bàn giao\n\n- Dataset được chọn cho issue này có hai nhãn AIRTLab: `non-violent` → `normal`; `violent` → `violence`.\n- Dataset hiện tại không có video fall; không dùng nó làm ground truth để huấn luyện hoặc đánh giá ngã.\n- Chưa nghiệm thu nhãn thủ công hoặc nhãn thời gian.\n- Chạy lại `download`, `prepare`, `audit`; sau đó kiểm tra các ứng viên gần trùng và cập nhật phiếu bằng kết quả thực tế.\n""",
        "docs/data_workflow.md": """# Thực hiện issue #1 — Dữ liệu\n\n## Chuẩn hóa video có sẵn trong `data/raw`\n\nNếu đã có video trong `data/raw` và chỉ cần chuẩn hóa để YOLO xử lý, chạy từ thư mục gốc:\n\n```powershell\npython -m src.data.pipeline normalize-raw\n```\n\nPipeline giữ cấu trúc thư mục, chuẩn hóa thành MP4 640×360 ở 15 FPS, không âm thanh, và không thay đổi video gốc. Chế độ này không tự suy đoán nhãn hoặc tạo split.\n\n## Tải và chuẩn bị dataset AIRTLab\n\nPhiên bản catalog này chỉ dùng 350 video AIRTLab. Nhãn gốc là `non-violent` và `violent`; dự án ánh xạ chúng lần lượt thành `normal` và `violence`. Đây là dataset hai lớp và không chứa video fall. Mô-đun phát hiện ngã trong dự án không đồng nghĩa với việc dataset này có mẫu ngã.\n\n```powershell\npython -m pip install -r requirements-data.txt\npython -m src.data.pipeline download --dataset airtlab\npython -m src.data.pipeline prepare\npython -m src.data.pipeline audit\npython -m src.data.report\n```\n\nChừa dung lượng cho 350 video gốc và các bản chuẩn hóa. Có retry, file `.part`, kiểm checksum AIRTLab, cache và giới hạn mặc định 64 MiB/file. Dùng `--workers 2` để giảm tải máy; mặc định là 4.\n\nKết quả gồm `data/raw/{normal,violence}`, `data/processed/{normal,violence}`, manifest và downloads trong `data/metadata/local`, cùng audit và thống kê dưới `docs/reports/`. Tổng byte, thời lượng và số frame phải được tạo lại sau khi chạy trên các file thực tế.\n\n## Chia tập, kiểm tra và nghiệm thu\n\nPipeline chia theo nhóm sự kiện, liên kết bản trùng và ứng viên gần trùng; không chia frame. Đọc cảnh báo audit và xem lại các ứng viên trước khi nghiệm thu. Báo cáo phải mô tả đúng hạn chế của dataset: video được diễn trong một phòng, không đại diện riêng cho lớp học.\n\nNếu cần huấn luyện/đánh giá lớp fall, cần bổ sung một nguồn được phê duyệt và lập phiên bản dataset mới có nguồn, nhãn, điều khoản và kiểm tra riêng. Không suy diễn lớp fall từ dữ liệu AIRTLab.\n""",
        "ISSUE_1_STATUS.md": """# Bàn giao issue #1 — Dữ liệu\n\nCatalog đã được giới hạn ở 350 video AIRTLab: 120 video nhãn gốc `non-violent` và 230 video nhãn gốc `violent`. Nhãn nội bộ được ánh xạ rõ ràng: `non-violent` → `normal`, `violent` → `violence`. Dataset hiện tại có hai lớp và không có video fall.\n\nSau khi thay đổi catalog, cần chạy lại tải, chuẩn hóa và audit để tái tạo checksum, thống kê media và báo cáo. Các số liệu trước thay đổi không còn đại diện cho catalog hiện tại.\n\nTrạng thái kiểm duyệt nhãn: chờ nhóm xem xét. Chưa có ground truth thời gian được nghiệm thu. Không mô tả dữ liệu AIRTLab là video lớp học thật hoặc dữ liệu fall.\n""",
    }
    changes = [f"rewrite {path} to describe only the selected AIRTLab dataset" for path in files]
    for relative, content in files.items():
        if apply:
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            atomic_write(target, content)
    readme = root / "README.md"
    if readme.exists():
        content = readme.read_text(encoding="utf-8")
        content = content.replace("- `data/raw/` - Chứa video gốc theo các lớp `normal`, `violence`, `fall`.",
                                  "- `data/raw/` - Video gốc; catalog AIRTLab hiện có hai nhãn nội bộ `normal` và `violence`.")
        content = content.replace("Phân loại ba trạng thái: `normal`, `violence`, `fall`.",
                                  "Mục tiêu dự án gồm nhận diện `normal`, `violence` và fall; catalog AIRTLab hiện chỉ cung cấp dữ liệu cho hai lớp đầu.")
        if apply:
            atomic_write(readme, content)
        changes.append("document the AIRTLab label scope in README.md")
    guide = root / "docs/annotation_guide.md"
    if guide.exists():
        content = guide.read_text(encoding="utf-8")
        mapping = ("Trong catalog AIRTLab, nhãn nguồn `non-violent` được ánh xạ thành `normal`, "
                   "còn `violent` được ánh xạ thành `violence`. AIRTLab không cung cấp ví dụ cho nhãn fall; "
                   "quy tắc fall bên dưới chỉ áp dụng khi có nguồn dữ liệu fall riêng đã được phê duyệt.")
        if content.startswith("# Hướng dẫn gán nhãn") and "Trong catalog AIRTLab" not in content:
            first, _, rest = content.partition("\n")
            content = first + "\n\n" + mapping + "\n\n" + rest
        if apply:
            atomic_write(guide, content)
        changes.append("document source-label mapping in docs/annotation_guide.md")
    return changes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="repository root")
    parser.add_argument("--apply", action="store_true", help="write changes; without this flag, show the plan only")
    args = parser.parse_args()
    root = args.root.resolve()
    if not (root / ".git").exists() or not (root / "data/metadata/catalog.jsonl").exists():
        parser.error("--root must point to the repository checkout")
    changes = migrate(root, args.apply) + clean_docs(root, args.apply)
    if args.apply:
        print(f"Applied {len(changes)} cleanup actions.")
        backup_dir = Path(tempfile.gettempdir()) / "classroom-violence-fall-detection-backups"
        print(f"Local source media was moved, and original local manifests were copied, to {backup_dir}.")
    else:
        print("Dry run. Re-run with --apply to make these changes:")
        for change in changes:
            print(f"- {change}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
