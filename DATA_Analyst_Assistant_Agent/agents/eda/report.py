import json
import os
import base64


def _img_tag(path: str) -> str:
    """이미지 파일을 base64로 인코딩해 <img> 태그 반환."""
    if not os.path.exists(path):
        return f'<p style="color:gray;">[파일 없음: {os.path.basename(path)}]</p>'
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f'<img src="data:image/png;base64,{b64}" style="max-width:100%;border-radius:6px;margin:8px 0;">'


def _text_block(text: str) -> str:
    """줄바꿈 보존해서 <p> 태그로 변환."""
    lines = text.strip().split("\n")
    result = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            result.append("<br>")
        else:
            result.append(f"<p>{stripped}</p>")
    return "\n".join(result)


def _insight_block(text: str) -> str:
    """insight_result를 섹션별로 구조화해서 렌더링."""
    import re

    section_icons = {
        "핵심 패턴": "📌",
        "구조 해석": "🔍",
        "해석 주의사항": "⚠️",
    }

    # 섹션 분리: [섹션명] 기준
    parts = re.split(r"\[([^\]]+)\]", text.strip())
    # parts = ['앞텍스트', '섹션명', '내용', '섹션명', '내용', ...]

    html = ""
    i = 1
    while i < len(parts) - 1:
        section_name = parts[i].strip()
        content = parts[i + 1].strip()
        icon = section_icons.get(section_name, "•")

        # 번호 붙은 항목을 카드로 변환
        items = re.split(r"\n(?=\d+\.)", content)
        cards_html = ""
        plain_html = ""

        for item in items:
            item = item.strip()
            if not item:
                continue
            num_match = re.match(r"^(\d+)\.\s*", item)
            if num_match:
                num = num_match.group(1)
                body = item[num_match.end():]
                cards_html += f"""
                <div class="insight-card">
                  <span class="insight-num">{num}</span>
                  <span class="insight-body">{body}</span>
                </div>"""
            else:
                # 번호 없는 줄 (예: 불릿 항목)
                for line in item.split("\n"):
                    line = line.strip().lstrip("- •")
                    if line:
                        plain_html += f'<p class="insight-plain">• {line}</p>'

        inner = (cards_html or "") + (plain_html or "")

        if section_name == "해석 주의사항":
            html += f"""
            <details class="insight-section caution">
              <summary>{icon} {section_name}</summary>
              <div class="insight-inner">{inner}</div>
            </details>"""
        else:
            html += f"""
            <div class="insight-section">
              <div class="insight-header">{icon} {section_name}</div>
              <div class="insight-inner">{inner}</div>
            </div>"""
        i += 2

    return html or _text_block(text)  # 파싱 실패 시 원본 그대로


def _hypothesis_block(text: str) -> str:
    """hypotheses를 [가설 N] 섹션별 카드로 구조화해서 렌더링."""
    import re

    parts = re.split(r"\[([^\]]+)\]", text.strip())
    # parts = ['앞텍스트', '가설 1', '내용', '가설 2', '내용', ..., '다음 분석 방향', '내용']

    html = ""
    hypothesis_cards = []
    next_direction_html = ""

    i = 1
    while i < len(parts) - 1:
        section_name = parts[i].strip()
        content = parts[i + 1].strip()

        if re.match(r"^가설\s*\d+$", section_name):
            # 가설 카드 파싱: 각 줄을 label: value로 분리
            label_icons = {
                "관찰": "🔎",
                "H0": "⚪",
                "H1": "🔵",
                "검증방법": "🧪",
                "필요변수": "📋",
                "현재데이터": "💾",
            }
            rows_html = ""
            for line in content.split("\n"):
                line = line.strip()
                if not line:
                    continue
                colon_idx = line.find(":")
                if colon_idx > 0:
                    label = line[:colon_idx].strip()
                    value = line[colon_idx + 1:].strip()
                    icon = label_icons.get(label, "•")
                    # H0/H1은 강조 스타일
                    if label in ("H0", "H1"):
                        rows_html += f'<div class="hyp-row hyp-hypothesis"><span class="hyp-label">{icon} {label}</span><span class="hyp-value">{value}</span></div>'
                    else:
                        rows_html += f'<div class="hyp-row"><span class="hyp-label">{icon} {label}</span><span class="hyp-value">{value}</span></div>'
                else:
                    rows_html += f'<div class="hyp-row"><span class="hyp-value">{line}</span></div>'

            hypothesis_cards.append(f"""
            <div class="hyp-card">
              <div class="hyp-title">{section_name}</div>
              {rows_html}
            </div>""")

        elif "다음 분석 방향" in section_name:
            lines_html = ""
            for line in content.split("\n"):
                line = line.strip()
                if not line:
                    continue
                lines_html += f'<p class="insight-plain">• {line.lstrip("0123456789. ")}</p>'
            next_direction_html = f"""
            <details class="insight-section caution">
              <summary>➡️ 다음 분석 방향</summary>
              <div class="insight-inner">{lines_html}</div>
            </details>"""

        i += 2

    if hypothesis_cards:
        html += '<div class="hyp-grid">' + "".join(hypothesis_cards) + "</div>"
    if next_direction_html:
        html += next_direction_html

    return html or _text_block(text)


def generate_report(json_path: str, output_html: str = None) -> str:
    """
    eda_agent_output.json을 읽어 HTML 리포트 생성.

    Args:
        json_path: eda_agent_output.json 경로
        output_html: 저장할 HTML 경로 (없으면 json_path 옆에 report.html)

    Returns:
        저장된 HTML 파일 경로
    """
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    if output_html is None:
        output_html = os.path.join(os.path.dirname(json_path), "report.html")

    user_question       = data.get("user_question", "")
    analysis_plan       = data.get("analysis_plan", {})
    insight_result      = data.get("insight_result", "")
    hypotheses          = data.get("hypotheses", "")
    final_summary       = data.get("final_summary", "")
    stat                = data.get("statistical_metadata", {})
    key_charts          = data.get("key_charts", [])

    # ── priority_metrics 테이블 ──
    priority_rows = ""
    for m in analysis_plan.get("priority_metrics", []):
        priority_rows += f"<tr><td>{m.get('metric','')}</td><td>{m.get('reason','')}</td></tr>"

    # ── 분석 계획 행 ──
    plan_flags = {
        "quality": analysis_plan.get("run_quality"),
        "distribution": analysis_plan.get("run_distribution"),
        "comparison": analysis_plan.get("run_comparison"),
        "relationship": analysis_plan.get("run_relationship"),
    }
    flag_chips = " ".join(
        f'<span class="chip {"on" if v else "off"}">{k}</span>'
        for k, v in plan_flags.items()
    )
    skip_reason = analysis_plan.get("skip_reason", "")

    # ── 통계 메타데이터 테이블 ──
    stat_rows = ""
    dist = stat.get("distribution", {})
    outliers = stat.get("outliers_by_column", {})
    for col, info in dist.items():
        stat_rows += f"""
        <tr>
            <td>{col}</td>
            <td>{info.get('mean',''):.4f}</td>
            <td>{info.get('median',''):.4f}</td>
            <td>{info.get('std',''):.4f}</td>
            <td>{info.get('skewness',''):.4f}</td>
            <td>{info.get('min','')}</td>
            <td>{info.get('max','')}</td>
            <td>{outliers.get(col, '-')}</td>
        </tr>"""

    corr_rows = ""
    for pair, val in stat.get("correlation_pairs", {}).items():
        label = pair.replace("corr_", "").replace("_vs_", " vs ")
        strength = abs(val)
        color = "#c0392b" if strength >= 0.3 else "#e67e22" if strength >= 0.1 else "#888"
        corr_rows += f'<tr><td>{label}</td><td style="color:{color};font-weight:bold;">{val:.3f}</td></tr>'

    all_charts          = data.get("all_charts", [])

    # ── 핵심 차트 ──
    key_set = {os.path.basename(p) for p in key_charts}
    chart_html = ""
    for path in key_charts:
        name = os.path.basename(path)
        chart_html += f"""
        <div class="chart-card">
            <div class="chart-name">{name}</div>
            {_img_tag(path)}
        </div>"""

    # ── 나머지 전체 차트 (핵심 차트 제외) ──
    all_chart_html = ""
    for path in all_charts:
        name = os.path.basename(path)
        if name in key_set:
            continue
        all_chart_html += f"""
        <div class="chart-card">
            <div class="chart-name">{name}</div>
            {_img_tag(path)}
        </div>"""

    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<title>EDA Report</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Segoe UI', sans-serif; background: #f5f6fa; color: #2c3e50; font-size: 14px; line-height: 1.7; }}
  .container {{ max-width: 960px; margin: 40px auto; padding: 0 20px; }}
  h1 {{ font-size: 22px; color: #1a252f; margin-bottom: 6px; }}
  .question-box {{ background: #2c3e50; color: #ecf0f1; padding: 16px 20px; border-radius: 8px; margin-bottom: 30px; font-size: 15px; }}
  section {{ background: white; border-radius: 8px; padding: 24px; margin-bottom: 24px; box-shadow: 0 1px 4px rgba(0,0,0,.08); }}
  section h2 {{ font-size: 15px; font-weight: 700; color: #2980b9; border-bottom: 2px solid #eaf2fb; padding-bottom: 8px; margin-bottom: 16px; text-transform: uppercase; letter-spacing: .5px; }}
  p {{ margin-bottom: 6px; }}
  .chip {{ display: inline-block; padding: 2px 10px; border-radius: 12px; font-size: 12px; margin-right: 6px; }}
  .chip.on  {{ background: #d5f5e3; color: #1e8449; }}
  .chip.off {{ background: #f2f3f4; color: #999; text-decoration: line-through; }}
  .skip {{ font-size: 12px; color: #888; margin-top: 8px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  th {{ background: #f0f4f8; text-align: left; padding: 8px 10px; font-weight: 600; }}
  td {{ padding: 7px 10px; border-bottom: 1px solid #f0f0f0; }}
  tr:hover td {{ background: #fafbfc; }}
  .summary-box {{ background: #eaf4fb; border-left: 4px solid #2980b9; padding: 14px 18px; border-radius: 4px; font-size: 14px; }}
  .charts-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(420px, 1fr)); gap: 20px; }}
  .chart-card {{ background: #f8f9fa; border-radius: 6px; padding: 12px; }}
  .chart-name {{ font-size: 12px; color: #888; margin-bottom: 6px; }}
  .meta-row {{ display: flex; gap: 20px; margin-bottom: 12px; font-size: 13px; }}
  .meta-item {{ background: #f0f4f8; border-radius: 6px; padding: 8px 14px; }}
  .meta-item span {{ font-weight: 700; font-size: 16px; display: block; }}
  details summary {{ cursor: pointer; font-size: 15px; font-weight: 700; color: #2980b9; padding: 16px 24px; background: white; border-radius: 8px; box-shadow: 0 1px 4px rgba(0,0,0,.08); margin-bottom: 4px; list-style: none; }}
  details summary::before {{ content: "▶ "; font-size: 11px; }}
  details[open] summary::before {{ content: "▼ "; }}
  details .inner {{ background: white; border-radius: 0 0 8px 8px; padding: 20px 24px; box-shadow: 0 1px 4px rgba(0,0,0,.08); margin-bottom: 24px; }}
  .insight-section {{ margin-bottom: 16px; }}
  .insight-header {{ font-weight: 700; font-size: 13px; color: #555; margin-bottom: 10px; }}
  .insight-inner {{ display: flex; flex-direction: column; gap: 8px; }}
  .insight-card {{ display: flex; gap: 12px; background: #f8f9fa; border-radius: 6px; padding: 10px 14px; align-items: flex-start; }}
  .insight-num {{ background: #2980b9; color: white; border-radius: 50%; width: 22px; height: 22px; display: flex; align-items: center; justify-content: center; font-size: 11px; font-weight: 700; flex-shrink: 0; margin-top: 2px; }}
  .insight-body {{ font-size: 13px; line-height: 1.6; color: #2c3e50; }}
  .insight-plain {{ font-size: 13px; color: #555; margin: 2px 0; }}
  details.insight-section {{ margin-bottom: 0; }}
  details.insight-section summary {{ font-size: 13px; font-weight: 700; color: #888; padding: 8px 0; background: none; border-radius: 0; box-shadow: none; border-top: 1px solid #eee; margin-top: 8px; }}
  details.insight-section summary::before {{ content: "▶ "; font-size: 10px; }}
  details.insight-section[open] summary::before {{ content: "▼ "; }}
  details.insight-section .insight-inner {{ margin-top: 8px; }}
  .hyp-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 16px; margin-bottom: 16px; }}
  .hyp-card {{ background: #f8f9fa; border-radius: 8px; padding: 16px; border-top: 3px solid #2980b9; }}
  .hyp-title {{ font-weight: 700; font-size: 14px; color: #2980b9; margin-bottom: 12px; }}
  .hyp-row {{ display: flex; gap: 8px; padding: 5px 0; border-bottom: 1px solid #eee; font-size: 13px; line-height: 1.5; }}
  .hyp-row:last-child {{ border-bottom: none; }}
  .hyp-label {{ font-weight: 600; color: #555; min-width: 80px; flex-shrink: 0; }}
  .hyp-value {{ color: #2c3e50; }}
  .hyp-hypothesis .hyp-label {{ color: #2980b9; }}
  .hyp-hypothesis .hyp-value {{ font-style: italic; }}
</style>
</head>
<body>
<div class="container">

  <h1>EDA Agent Report</h1>
  <div class="question-box">{user_question}</div>

  <!-- 분석 계획 -->
  <section>
    <h2>Analysis Plan</h2>
    <div style="margin-bottom:12px;">{flag_chips}</div>
    {f'<div class="skip">{skip_reason}</div>' if skip_reason else ''}
    {f'''
    <table style="margin-top:14px;">
      <tr><th>우선 지표</th><th>선정 이유</th></tr>
      {priority_rows}
    </table>''' if priority_rows else ''}
  </section>

  <!-- 통계 메타데이터 -->
  <section>
    <h2>Statistical Metadata</h2>
    <div class="meta-row">
      <div class="meta-item">행 수<span>{stat.get('row_count', '-'):,}</span></div>
      <div class="meta-item">결측치<span>{stat.get('missing_total', '-')}</span></div>
      <div class="meta-item">중복<span>{stat.get('duplicate_count', '-')}</span></div>
    </div>
    <table>
      <tr><th>컬럼</th><th>mean</th><th>median</th><th>std</th><th>skewness</th><th>min</th><th>max</th><th>이상치</th></tr>
      {stat_rows}
    </table>
    {f'''
    <table style="margin-top:16px;">
      <tr><th>상관관계 쌍</th><th>r</th></tr>
      {corr_rows}
    </table>''' if corr_rows else ''}
  </section>

  <!-- 인사이트 -->
  <section>
    <h2>Insight</h2>
    {_insight_block(insight_result)}
  </section>

  <!-- 가설 -->
  <section>
    <h2>Hypotheses</h2>
    {_hypothesis_block(hypotheses)}
  </section>

  <!-- 최종 요약 -->
  <section>
    <h2>Final Summary</h2>
    <div class="summary-box">{final_summary}</div>
  </section>

  <!-- 핵심 차트 -->
  <section>
    <h2>Key Charts ({len(key_charts)})</h2>
    <div class="charts-grid">
      {chart_html}
    </div>
  </section>

  <!-- 전체 차트 (접기/펼치기) -->
  {f'''
  <details>
    <summary>All Charts ({len(all_charts) - len(key_charts)} 나머지 / 전체 {len(all_charts)})</summary>
    <div class="inner">
      <div class="charts-grid">
        {all_chart_html}
      </div>
    </div>
  </details>
  ''' if all_chart_html else ''}

</div>
</body>
</html>"""

    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html)

    return output_html
