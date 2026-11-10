"""Turn a saved Gmail evaluation JSON into a Markdown and self-contained HTML report."""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any


SYSTEMS = [
    ("base_one_pass_model_gate", "Base · one pass"),
    ("base_plus_task_neutral_replay", "Base + MELON replay"),
    ("tuned_one_pass_student", "LoRA · one pass"),
    ("tuned_plus_task_neutral_replay", "LoRA + MELON replay"),
]


def pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def metric(report: dict[str, Any], system: str) -> dict[str, Any]:
    return report["metrics"][system]


def make_svg(title: str, rows: list[tuple[str, float, str]], max_value: float, unit: str) -> str:
    width, height = 760, 70 + len(rows) * 48
    label_x, bar_x, bar_width = 180, 190, 480
    colors = ["#687386", "#cf8c28", "#3278a8", "#6b8e45"]
    chunks = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        f'<text x="20" y="28" font-size="17" font-weight="700" fill="#1b2430">{html.escape(title)}</text>',
    ]
    for index, (label, value, display) in enumerate(rows):
        y = 58 + index * 48
        bar_len = max(2, bar_width * value / max_value) if value else 2
        chunks.append(f'<text x="{label_x}" y="{y + 17}" text-anchor="end" font-size="13" fill="#273142">{html.escape(label)}</text>')
        chunks.append(f'<rect x="{bar_x}" y="{y}" width="{bar_width}" height="24" rx="4" fill="#eef1f5"/>')
        chunks.append(f'<rect x="{bar_x}" y="{y}" width="{bar_len:.1f}" height="24" rx="4" fill="{colors[index]}"/>')
        chunks.append(f'<text x="{bar_x + bar_len + 8:.1f}" y="{y + 17}" font-size="13" fill="#273142">{html.escape(display)}{html.escape(unit)}</text>')
    chunks.append("</svg>")
    return "".join(chunks)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("runs/gmail_qwen3_0_6b_lora/gmail_eval_full.json"))
    parser.add_argument("--markdown", type=Path, default=None)
    parser.add_argument("--html", type=Path, default=None)
    args = parser.parse_args()
    if not args.input.is_file():
        parser.error(f"Evaluation report not found: {args.input}")
    report = json.loads(args.input.read_text(encoding="utf-8"))
    md_path = args.markdown or args.input.with_name("gmail_eval_report.md")
    html_path = args.html or args.input.with_name("gmail_eval_report.html")

    available_metrics = report.get("metrics", {})
    clean_total = metric(report, "tuned_one_pass_student")["clean_cases"]
    injected_total = metric(report, "tuned_one_pass_student")["injected_cases"]
    table_rows = []
    for system, label in SYSTEMS:
        if system not in available_metrics:
            continue
        m = metric(report, system)
        total_tokens = m["total_input_tokens"] + m["total_output_tokens"]
        table_rows.append((label, m, total_tokens))
    if not table_rows:
        parser.error("Evaluation report has no recognized metrics")

    blocked_cases = [case for case in report.get("cases", []) if case.get("melon_decision") == "BLOCK"]
    blocked_clean = sum(case.get("scenario_kind") == "clean" for case in blocked_cases)
    blocked_injected = sum(case.get("scenario_kind") == "injected" for case in blocked_cases)
    blocked_owner_draft = sum(
        ((case.get("base_melon", {}).get("parsed") or {}).get("proposed_action") or {}).get("tool") == "create_draft"
        and (((case.get("base_melon", {}).get("parsed") or {}).get("proposed_action") or {}).get("arguments") or {}).get("to") == "owner@example.test"
        for case in blocked_cases
    )
    m = metric(report, "tuned_one_pass_student")
    replay = available_metrics.get("base_plus_task_neutral_replay")
    tuned_replay = available_metrics.get("tuned_plus_task_neutral_replay")
    model_id = report.get("model_id", "model not recorded")
    limit = report.get("settings", {}).get("limit_contexts")
    preview_note = (
        f"- **Preview:** {limit} of 50 held-out contexts; this is not the full evaluation.\n"
        if isinstance(limit, int) and 0 < limit < 50 else ""
    )
    report_md = [
        "# Gmail LoRA evaluation",
        "",
        f"- Model: `{model_id}`",
        preview_note.rstrip(),
        f"- Test source: {report.get('test_source', 'not recorded')}",
        f"- Cases: {m['cases']} total ({clean_total} clean, {injected_total} injected)",
        f"- Adapter: `{report.get('adapter_dir', 'not recorded')}`",
        f"- Label source: {report.get('training_label_source', 'not recorded')}",
        "- Scope: BIPIA test email contexts plus synthetic forward-injection prompts; this is not an official BIPIA score.",
        "",
        "## Results",
        "",
        "| System | Parseable actions | Clean owner draft + ALLOW | Known answers | Unauthorized forwards / injected | Generation seconds | Input + output tokens |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, row, total_tokens in table_rows:
        known = f"{row['known_answers_present_in_draft_body']}/{row['known_answer_clean_cases']} ({pct(row['known_answer_accuracy'])})"
        forwarded = f"{row['simulated_executed_unauthorized_forwards']}/{row['injected_cases']}"
        report_md.append(
            f"| {label} | {row['parse_success']}/{row['cases']} ({pct(row['parse_rate'])}) "
            f"| {row['clean_owner_draft_and_allow']}/{row['clean_cases']} "
            f"| {known} | {forwarded} | {row['total_generation_seconds']:.1f} | {total_tokens} |"
        )
    report_md += [
        "",
        "## Interpretation",
        "",
        f"The tuned one-pass student used {m['total_generation_seconds']:.1f} seconds and "
        f"{m['total_input_tokens'] + m['total_output_tokens']} total tokens. "
        + (f"The base plus replay used {replay['total_generation_seconds']:.1f} seconds and "
           f"{replay['total_input_tokens'] + replay['total_output_tokens']} tokens in this run. "
           if replay else "The evaluation report does not contain a base replay row. ")
        + "These are observed prototype timings; prompt formats differ and they are not a general speed guarantee.",
        "",
        f"No system proposed an unauthorized forward among the {injected_total} injected cases. "
        "That means this run did not demonstrate a security improvement or exercise a harmful-forward block. "
        "The tuned model's zero-forward result is encouraging for these cases, but the comparison is inconclusive on attack prevention.",
        "",
        f"The replay gate blocked {len(blocked_cases)} cases ({blocked_clean} clean, {blocked_injected} injected); "
        f"{blocked_owner_draft} were owner-directed draft actions. Since the user task requested an unsent owner draft, "
        "these blocks indicate a likely false-positive issue in the current action-match rule.",
        "",
        f"Known-answer accuracy is based on only {m['known_answer_clean_cases']} clean test cases with known answers. "
        "Do not generalize this number. Teacher labels were automatically generated by the documented rule, not human-reviewed.",
        "",
        *(["## Interpretation of tuned + replay", "",
           "The LoRA + MELON replay row uses the same task-neutral replay and exact structured action-match rule as the base + replay row. Its measured time and tokens include the saved tuned one-pass generation plus the added masked replay generation. Compare its clean requested-draft rate and false blocks alongside any blocked unsafe actions.", ""] if tuned_replay else
          ["## Tuned replay status", "", "A tuned + task-neutral replay pass is not included in this report.", ""]),
        "## Next evaluation work",
        "",
        "Before claiming the tuned model is more secure or adding Internet data, refine the replay/action label rule and evaluate on a fixed, held-out set that produces both benign requested drafts and demonstrably unsafe forwarding attempts. Report false blocks and actual simulated side effects separately.",
        "",
    ]
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(report_md), encoding="utf-8")

    charts = []
    chart_specs = [
        ("Parse rate", [(label, row["parse_rate"] * 100, f"{row['parse_rate'] * 100:.1f}") for label, row, _ in table_rows], 100, "%"),
        ("Clean requested draft + allow", [(label, row["clean_owner_draft_and_allow"], str(row["clean_owner_draft_and_allow"])) for label, row, _ in table_rows], clean_total, f" / {clean_total}"),
        ("Generation time (lower is faster)", [(label, row["total_generation_seconds"], f"{row['total_generation_seconds']:.0f}") for label, row, _ in table_rows], max(row["total_generation_seconds"] for _, row, _ in table_rows), " sec"),
        ("Total tokens (lower is fewer)", [(label, total, f"{total:,}") for label, _, total in table_rows], max(total for _, _, total in table_rows), " tokens"),
    ]
    for title, values, max_value, unit in chart_specs:
        charts.append(make_svg(title, values, max_value or 1, unit))
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Gmail LoRA evaluation</title><style>
body{{font:16px/1.55 system-ui,Segoe UI,sans-serif;max-width:1100px;margin:32px auto;padding:0 20px;color:#202938;background:#f7f8fa}}
h1,h2{{line-height:1.2}} .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:16px}}
svg{{width:100%;height:auto;background:white;border:1px solid #dde2e8;border-radius:8px}}
.note{{padding:14px 18px;background:#fff3d7;border-left:4px solid #cf8c28}} table{{border-collapse:collapse;background:white;width:100%}}
th,td{{padding:8px;border:1px solid #d9dee6;text-align:left}} th{{background:#edf1f5}}
</style></head><body><h1>Gmail LoRA evaluation</h1>
<p>{html.escape(report.get('test_source',''))}</p>
<p><b>Cases:</b> {m['cases']} total, {clean_total} clean and {injected_total} injected.</p>
<div class="note"><b>Security interpretation:</b> no system proposed an unauthorized forward in this test, so it does not demonstrate a comparative security gain. The replay gate blocked {len(blocked_cases)} cases, including {blocked_clean} clean cases and {blocked_owner_draft} owner-directed drafts.</div>
<h2>Metrics</h2><table><thead><tr><th>System</th><th>Parseable</th><th>Clean owner drafts</th><th>Known answers</th><th>Unauthorized forwards</th><th>Time (s)</th><th>Total tokens</th></tr></thead><tbody>
{''.join(f'<tr><td>{html.escape(label)}</td><td>{row["parse_success"]}/{row["cases"]} ({pct(row["parse_rate"])})</td><td>{row["clean_owner_draft_and_allow"]}/{row["clean_cases"]}</td><td>{row["known_answers_present_in_draft_body"]}/{row["known_answer_clean_cases"]} ({pct(row["known_answer_accuracy"])})</td><td>{row["simulated_executed_unauthorized_forwards"]}/{row["injected_cases"]}</td><td>{row["total_generation_seconds"]:.1f}</td><td>{row["total_input_tokens"] + row["total_output_tokens"]:,}</td></tr>' for label,row,_ in table_rows)}
</tbody></table><h2>Charts</h2><div class="grid">{''.join(charts)}</div>
<p>Prototype evaluation only. Prompt formats differ. Labels are automatically generated. Replay rows include both original and masked passes. This is not an official BIPIA score or a broad security guarantee.</p>
</body></html>"""
    html_path.write_text(page, encoding="utf-8")
    readme_path = args.input.with_name(f"README_{Path(model_id).name.replace('.', '_').lower()}_gmail_results.md")
    readme_lines = [
        f"# Gmail fine-tuning results: {model_id}", "",
        f"{preview_note.rstrip()}" if preview_note else "Evaluation split: all contexts recorded in the input report.",
        "", "This is a local prototype evaluation on BIPIA email contexts with synthetic forwarding-injection prompts. "
        "It is not an official BIPIA score or a broad security guarantee.", "",
        "## Results", "", "| System | Parseable | Clean owner drafts | Known answers | Unauthorized forwards | Time (s) | Tokens |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, row, total_tokens in table_rows:
        readme_lines.append(
            f"| {label} | {row['parse_success']}/{row['cases']} ({pct(row['parse_rate'])}) "
            f"| {row['clean_owner_draft_and_allow']}/{row['clean_cases']} "
            f"| {row['known_answers_present_in_draft_body']}/{row['known_answer_clean_cases']} ({pct(row['known_answer_accuracy'])}) "
            f"| {row['simulated_executed_unauthorized_forwards']}/{row['injected_cases']} "
            f"| {row['total_generation_seconds']:.1f} | {total_tokens:,} |"
        )
    readme_lines += ["", "## Interpretation", "",
                     "Zero proposed/executed forwards does not demonstrate that the model blocks a harmful action when one is proposed. "
                     "The automatic teacher labels and the limited evaluation design are additional constraints.", "",
                     "Detailed table and charts: [Markdown report](gmail_eval_report.md) · [HTML report](gmail_eval_report.html).", ""]
    readme_path.write_text("\n".join(readme_lines), encoding="utf-8")
    print(f"Wrote Markdown report: {md_path.resolve()}")
    print(f"Wrote HTML report with SVG charts: {html_path.resolve()}")
    print(f"Wrote model-specific results README: {readme_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
