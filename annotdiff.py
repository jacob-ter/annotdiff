"""annotdiff —— 比对两批标注结果，定位分歧样本。

设计原则：
1. **一致率不告诉你在哪分歧**。这个工具列出具体分歧样本，
   以及分歧最集中的类别。
2. **分歧指向规则，不是指向人**。输出的用途是「该改哪条规则」。
3. **不同一致率算法都算一遍**。算法选不同，结论会差很多。

仅依赖 Python 标准库。

用法:
    python annotdiff.py --help
    python annotdiff.py --a a.jsonl --b b.jsonl
    python annotdiff.py --a a.jsonl --b b.jsonl --field label --json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict

# ---------------------------------------------------------------------------
# 输入
# ---------------------------------------------------------------------------

def load_annotations(path: str) -> list[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read().strip()
    except OSError as e:
        raise ValueError(f"读不了 {path}：{e}") from e
    if not raw:
        return []
    if raw.startswith("["):
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON 解析失败：{e}") from e
        return [x for x in data if isinstance(x, dict)]

    rows: list[dict] = []
    for lineno, line in enumerate(raw.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"第 {lineno} 行 JSON 解析失败：{e}") from e
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def _get(rec: dict, field: str):
    cur = rec
    for part in field.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def _norm_label(v) -> str:
    """把标签归一成可比较的字符串。"""
    if v is None:
        return ""
    if isinstance(v, (list, tuple, set)):
        items = sorted(str(x) for x in v)
        return "|".join(items)
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v).strip()


# ---------------------------------------------------------------------------
# 一致率
# ---------------------------------------------------------------------------

def exact_agreement(pairs: list[tuple[str, str]]) -> float:
    """完全一致率。"""
    if not pairs:
        return 0.0
    same = sum(1 for a, b in pairs if a == b)
    return same / len(pairs)


def cohen_kappa(pairs: list[tuple[str, str]]) -> dict:
    """Cohen's kappa。"""
    n = len(pairs)
    if n == 0:
        return {"kappa": None, "po": 0.0, "pe": None}
    same = sum(1 for a, b in pairs if a == b)
    po = same / n

    labels_a = Counter(a for a, _ in pairs)
    labels_b = Counter(b for _, b in pairs)
    pe = sum((labels_a[l] / n) * (labels_b[l] / n) for l in set(labels_a) | set(labels_b))

    kappa = (po - pe) / (1 - pe) if pe < 1 else 1.0
    return {"kappa": round(kappa, 4), "po": round(po, 4),
            "pe": round(pe, 4)}


def jaccard_agreement(pairs: list[tuple[str, str]]) -> float:
    """多标签场景的 Jaccard 一致率。"""
    if not pairs:
        return 0.0
    total = 0.0
    for a, b in pairs:
        sa = set(a.split("|")) if a else set()
        sb = set(b.split("|")) if b else set()
        if not sa and not sb:
            total += 1.0
            continue
        union = sa | sb
        total += (len(sa & sb) / len(union)) if union else 1.0
    return total / len(pairs)


# ---------------------------------------------------------------------------
# 分歧定位
# ---------------------------------------------------------------------------

def find_divergences(a_rows: list[dict], b_rows: list[dict],
                     id_field: str, field: str,
                     category_field: str) -> dict:
    """按样本比对两批标注，定位分歧。"""
    def index(rows: list[dict]) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for i, r in enumerate(rows):
            rid = _get(r, id_field)
            key = str(rid) if rid is not None else f"__idx_{i}"
            out[key] = r
        return out

    ia, ib = index(a_rows), index(b_rows)

    common = sorted(set(ia) & set(ib))
    only_a = sorted(set(ia) - set(ib))
    only_b = sorted(set(ib) - set(ia))

    pairs: list[tuple[str, str]] = []
    divergences: list[dict] = []
    by_category: dict[str, Counter] = defaultdict(Counter)

    for key in common:
        la = _norm_label(_get(ia[key], field))
        lb = _norm_label(_get(ib[key], field))
        pairs.append((la, lb))

        if la != lb:
            cat = _get(ia[key], category_field)
            cat_key = str(cat) if cat is not None else "（无类别字段）"
            by_category[cat_key][f"{la} → {lb}"] += 1
            divergences.append({
                "id": key,
                "a": la,
                "b": lb,
                "category": cat_key,
                "a_text": str(_get(ia[key], "text") or "")[:80],
            })

    # 分歧最集中的类别
    category_rank = []
    for cat, counter in by_category.items():
        total = sum(counter.values())
        top, top_n = counter.most_common(1)[0]
        category_rank.append({
            "category": cat,
            "divergences": total,
            "share_of_total": round(total / len(divergences), 4) if divergences else 0,
            "top_confusion": f"{top}（{top_n} 次）",
        })
    category_rank.sort(key=lambda c: (-c["divergences"], c["category"]))

    # 哪些标签对最容易分歧
    pair_counts = Counter((d["a"], d["b"]) for d in divergences)
    confusion = [{"a": a, "b": b, "count": c}
                 for (a, b), c in pair_counts.most_common(20)]

    exact = exact_agreement(pairs)
    kappa = cohen_kappa(pairs)
    jac = jaccard_agreement(pairs)

    return {
        "common_count": len(common),
        "only_a": only_a,
        "only_b": only_b,
        "divergences": divergences,
        "divergence_count": len(divergences),
        "divergence_rate": round(len(divergences) / len(common), 4) if common else 0.0,
        "categories": category_rank,
        "confusion": confusion,
        "agreement": {
            "exact": round(exact, 4),
            "cohen_kappa": kappa["kappa"],
            "cohen_po": kappa["po"],
            "cohen_pe": kappa["pe"],
            "jaccard": round(jac, 4),
            "note": ("不同算法结论可能差很多。exact 最严格；"
                     "kappa 扣除了「碰巧一致」的部分；jaccard 适合多标签。"),
        },
        "label_counts_a": dict(Counter(a for a, _ in pairs)),
        "label_counts_b": dict(Counter(b for _, b in pairs)),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def render(result: dict) -> str:
    out: list[str] = []
    ag = result["agreement"]
    out.append("标注分歧比对")
    out.append("=" * 64)
    out.append(f"共有样本 {result['common_count']} 条 ｜ "
               f"分歧 {result['divergence_count']} 条 "
               f"（{result['divergence_rate']:.1%}）")
    out.append("")

    out.append("一致率（不同算法）：")
    out.append(f"  完全一致率      {ag['exact']:.4f}")
    out.append(f"  Cohen's kappa   {ag['cohen_kappa']}")
    out.append(f"    （观察一致 {ag['cohen_po']}，偶然一致 {ag['cohen_pe']}）")
    out.append(f"  Jaccard         {ag['jaccard']:.4f}")
    out.append(f"  {ag['note']}")
    out.append("")

    if result["only_a"] or result["only_b"]:
        out.append(f"只在 A 里的样本：{len(result['only_a'])} 条")
        out.append(f"只在 B 里的样本：{len(result['only_b'])} 条")
        out.append("")

    if result["categories"]:
        out.append("分歧最集中的类别：")
        for c in result["categories"][:10]:
            out.append(f"  {c['category']:<24} 分歧 {c['divergences']:>4} 条"
                       f"（占全部分歧 {c['share_of_total']:.1%}）")
            out.append(f"      最常见：{c['top_confusion']}")
        out.append("")

    if result["confusion"]:
        out.append("最容易分歧的标签组合：")
        for c in result["confusion"][:10]:
            out.append(f"  {c['a']:<20} → {c['b']:<20} {c['count']} 次")
        out.append("")

    out.append("分歧样例：")
    for d in result["divergences"][:10]:
        out.append(f"  {d['id']}  [{d['category']}]")
        out.append(f"    A: {d['a']}    B: {d['b']}")
        if d["a_text"]:
            out.append(f"    文本：{d['a_text']}")
    out.append("")
    out.append("注：分歧集中说明该改标注规则，不是该给谁记过。")
    return "\n".join(out).rstrip()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="annotdiff",
        description="比对两批标注结果，定位分歧样本（仅依赖标准库）",
        epilog=(
            "标注文件（JSONL）：\n"
            "  {\"id\":\"s1\",\"text\":\"...\",\"category\":\"退款\",\"label\":\"通过\"}\n"
            "  python annotdiff.py --a a.jsonl --b b.jsonl --field label\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--a", required=True, help="A 批标注")
    ap.add_argument("--b", required=True, help="B 批标注")
    ap.add_argument("--field", default="label", help="标注字段")
    ap.add_argument("--id-field", default="id", help="样本 id 字段")
    ap.add_argument("--category-field", default="category",
                    help="类别字段（用来定位分歧集中处）")
    ap.add_argument("--json", action="store_true", help="输出 JSON")
    args = ap.parse_args(argv)

    try:
        a_rows = load_annotations(args.a)
        b_rows = load_annotations(args.b)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 2

    if not a_rows or not b_rows:
        print("有一批标注是空的。", file=sys.stderr)
        return 2

    result = find_divergences(a_rows, b_rows, args.id_field, args.field,
                               args.category_field)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    print(render(result))
    return 1 if result["divergence_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
