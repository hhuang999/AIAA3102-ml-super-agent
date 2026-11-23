"""Usage pages: a 3D donut for composition, moving bars for parts and hours.

The board shows one task at a time; a picker switches task. The page reloads
itself every five minutes, and a button reloads it immediately.
"""

from __future__ import annotations

_STYLE = """
  :root { color-scheme: dark; --ink:#f4f7fb; --muted:#93a3b8; --line:rgba(255,255,255,.09); }
  * { box-sizing:border-box; }
  html { scroll-behavior:smooth; }
  body {
    margin:0; min-height:100vh; color:var(--ink); font:15px/1.5 "Segoe UI","PingFang SC","Noto Sans SC",sans-serif;
    background:radial-gradient(1200px 700px at 12% -10%, #1b2a4a 0%, #0b1220 55%, #070a12 100%);
    -webkit-font-smoothing:antialiased;
  }
  body::before {
    content:""; position:fixed; inset:-20%; z-index:0; pointer-events:none;
    background:
      radial-gradient(420px 420px at 18% 22%, rgba(255,77,109,.22), transparent 62%),
      radial-gradient(460px 460px at 82% 12%, rgba(78,205,196,.20), transparent 62%),
      radial-gradient(520px 520px at 68% 88%, rgba(139,92,246,.20), transparent 62%);
    filter:blur(10px); animation:drift 26s ease-in-out infinite alternate;
  }
  @keyframes drift { from { transform:translate3d(-2%,-1%,0) scale(1); } to { transform:translate3d(3%,2%,0) scale(1.08); } }
  .wrap { position:relative; z-index:1; max-width:1140px; margin:0 auto; padding:30px 20px 84px; }
  header.bar { display:flex; justify-content:space-between; align-items:flex-end; gap:18px; flex-wrap:wrap; margin-bottom:20px; }
  h1 { font-size:clamp(24px,3.1vw,34px); letter-spacing:-.035em; margin:0; font-weight:750;
       background:linear-gradient(92deg,#fff 10%,#9fd8ff 52%,#ffb3c7 92%);
       -webkit-background-clip:text; background-clip:text; color:transparent; }
  .sub, .note { color:var(--muted); margin:7px 0 0; font-size:13px; }
  .note { margin-top:18px; }
  .actions { display:flex; gap:9px; align-items:center; flex-wrap:wrap; }
  select, button.reload { border-radius:14px; padding:10px 15px; font:inherit; backdrop-filter:blur(10px); }
  select { border:1px solid var(--line); background:rgba(255,255,255,.06); color:var(--ink); max-width:min(430px,74vw); }
  select option { background:#121a2b; color:var(--ink); }
  button.reload { border:0; cursor:pointer; color:#0a1020; font-weight:700;
    background:linear-gradient(96deg,#4ecdc4,#7dd3fc 55%,#c084fc); box-shadow:0 8px 24px rgba(78,205,196,.28);
    transition:transform .18s ease, box-shadow .18s ease; }
  button.reload:hover { transform:translateY(-2px); box-shadow:0 12px 30px rgba(124,196,255,.38); }
  button.reload:active { transform:translateY(0) scale(.98); }
  .stats { display:grid; grid-template-columns:repeat(4,1fr); gap:14px; margin-bottom:14px; }
  .stat, .card {
    border:1px solid var(--line); border-radius:22px; background:linear-gradient(160deg,rgba(255,255,255,.085),rgba(255,255,255,.025));
    box-shadow:0 18px 44px rgba(0,0,0,.32), inset 0 1px 0 rgba(255,255,255,.08);
  }
  .stat { padding:17px 19px; position:relative; overflow:hidden; }
  .stat::after { content:""; position:absolute; inset:auto -20% -60% auto; width:150px; height:150px;
    background:radial-gradient(circle,rgba(124,196,255,.24),transparent 70%); }
  .stat .k { color:var(--muted); font-size:12.5px; letter-spacing:.02em; }
  .stat .v { font-size:clamp(22px,2.6vw,30px); margin-top:5px; font-variant-numeric:tabular-nums; letter-spacing:-.03em; font-weight:700; }
  .stat .h { color:var(--muted); font-size:12px; margin-top:3px; }
  .grid { display:grid; grid-template-columns:minmax(0,1fr) minmax(0,1.05fr); gap:14px; }
  .card { padding:20px; }
  .card h2 { font-size:14px; margin:0 0 14px; font-weight:650; letter-spacing:.03em; text-transform:uppercase; color:#b9c8dc; }
  .stage { display:grid; place-items:center; padding:6px 0 2px; }
  .stage svg { max-width:100%; height:auto; overflow:visible; filter:drop-shadow(0 22px 30px rgba(0,0,0,.42)); }
  .slice { transform-box:fill-box; transform-origin:center; animation:rise .85s cubic-bezier(.22,1.1,.36,1) both; cursor:pointer;
           transition:opacity .2s ease, filter .2s ease; }
  .slice:hover { filter:brightness(1.14) saturate(1.1); }
  .slice text { paint-order:stroke; stroke:rgba(6,10,18,.42); stroke-width:2.4px; }
  @keyframes rise { from { opacity:0; transform:translateY(26px) scale(.94); } to { opacity:1; transform:none; } }
  .legend { list-style:none; margin:0; padding:0; display:grid; gap:8px; }
  .legend li { display:flex; align-items:center; gap:9px; font-size:13px; color:var(--muted); }
  .legend b { color:var(--ink); font-weight:650; }
  .sw { width:11px; height:11px; border-radius:4px; flex:none; box-shadow:0 0 0 3px rgba(255,255,255,.05); }
  .bars { display:grid; gap:10px; }
  .bar { display:grid; grid-template-columns:minmax(92px,150px) minmax(0,1fr) auto; gap:11px; align-items:center; font-size:13px; }
  .bar > span:first-child { color:#cfdcec; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
  .track { height:12px; border-radius:999px; background:rgba(255,255,255,.07); overflow:hidden; }
  .fill { display:block; height:100%; border-radius:999px; width:0; animation:grow 1.05s cubic-bezier(.22,1,.36,1) forwards;
          background-image:linear-gradient(90deg,rgba(255,255,255,.32),rgba(255,255,255,0)); background-blend-mode:overlay; }
  @keyframes grow { to { width:var(--w); } }
  .num { font-variant-numeric:tabular-nums; color:#9fb0c6; white-space:nowrap; }
  .empty { color:var(--muted); margin:8px 0 0; }
  .cols { display:flex; align-items:flex-end; gap:8px; height:150px; padding-top:6px; }
  .col { flex:1; min-width:0; display:flex; flex-direction:column; justify-content:flex-end; gap:6px; text-align:center; }
  .col .stem { border-radius:10px 10px 4px 4px; height:0; animation:growY 1s cubic-bezier(.22,1,.36,1) forwards;
               background:linear-gradient(180deg,#7dd3fc,#4ecdc4 55%,#3b82f6); box-shadow:0 0 18px rgba(78,205,196,.28); }
  @keyframes growY { to { height:var(--h); } }
  .col .cap { font-size:11px; color:var(--muted); white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
  .col .val { font-size:11px; color:#cfe0f2; font-variant-numeric:tabular-nums; }
  @media (max-width:900px) { .stats { grid-template-columns:repeat(2,1fr); } .grid { grid-template-columns:1fr; } }
  @media (max-width:560px) { .stats { grid-template-columns:1fr; } .bar { grid-template-columns:1fr auto; } .bar .track { grid-column:1/-1; } }
"""


import html
import math
from datetime import datetime
from decimal import Decimal
from typing import Any, Mapping, Sequence

_PALETTE = (
    "#ff4d6d", "#ff8c42", "#ffc93c", "#4ecdc4", "#3b82f6",
    "#8b5cf6", "#ec4899", "#14b8a6", "#f97316", "#a3e635",
)
_MIX = (("input", "#ff4d6d"), ("output", "#4ecdc4"), ("cache", "#a3e635"))
SEP = " \u00b7 "
_MIX_CN = {"input": "\u8f93\u5165", "output": "\u8f93\u51fa", "cache": "\u7f13\u5b58"}


def _esc(value: object) -> str:
    return html.escape(str(value if value is not None else ""))


def _money(value: object) -> str:
    try:
        return f"{Decimal(str(value or '0')):.2f}"
    except Exception:
        return "0.00"


def _compact(n: object) -> str:
    n = int(n or 0)
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 10_000:
        return f"{n / 1_000:.1f}k"
    return f"{n:,}"


def _shade(color: str, factor: float) -> str:
    raw = color.lstrip("#")
    try:
        r, g, b = (int(raw[i:i + 2], 16) for i in (0, 2, 4))
    except ValueError:
        return color
    return "#%02x%02x%02x" % (max(0, min(255, int(r * factor))),
                              max(0, min(255, int(g * factor))),
                              max(0, min(255, int(b * factor))))


def _front_segments(a0: float, a1: float) -> list[tuple[float, float]]:
    two_pi = 2 * math.pi
    start = a0 % two_pi
    span = a1 - a0
    if span >= two_pi:
        return [(0.0, math.pi)]
    end = start + span
    out = []
    k = math.floor(start / two_pi)
    while k * two_pi <= end:
        lo = max(start, k * two_pi)
        hi = min(end, k * two_pi + math.pi)
        if hi > lo:
            out.append((lo, hi))
        k += 1
    return out


def _pie3d(slices: Sequence[tuple[str, float, str]], *, width: int = 420, height: int = 320,
           depth: int = 30, explode: float = 0.06) -> str:
    usable = [(label, float(value), color) for label, value, color in slices if float(value) > 0]
    if not usable:
        return ("<svg viewBox='0 0 420 300' width='100%'><circle cx='210' cy='150' r='86' fill='none' "
                "stroke='rgba(255,255,255,.12)' stroke-width='30'/><text x='210' y='156' text-anchor='middle' "
                "fill='#7a8ba3' font-size='13'>no calls yet</text></svg>")
    total = sum(v for _, v, _ in usable)
    rx = min(width, height * 1.42) * 0.355
    ry = rx * 0.56
    cx, cy = width / 2.0, height / 2.0 - depth * 0.3
    spans, start = [], -math.pi / 2
    for label, value, color in usable:
        end = start + (value / total) * 2 * math.pi
        spans.append((label, value / total, color, start, end))
        start = end

    def pt(angle, ax, ay, sx, sy):
        return (cx + sx + ax * math.cos(angle), cy + sy + ay * math.sin(angle))

    walls, tops = [], []
    for index, (label, frac, color, a0, a1) in enumerate(spans):
        mid = (a0 + a1) / 2.0
        sx, sy = math.cos(mid) * rx * explode, math.sin(mid) * ry * explode
        span = a1 - a0
        flag = 1 if span > math.pi else 0
        dark = _shade(color, 0.55)
        for s0, s1 in _front_segments(a0, a1):
            x0, y0 = pt(s0, rx, ry, sx, sy)
            x1, y1 = pt(s1, rx, ry, sx, sy)
            f = 1 if (s1 - s0) > math.pi else 0
            path = (f"M {x0:.2f} {y0:.2f} A {rx:.2f} {ry:.2f} 0 {f} 1 {x1:.2f} {y1:.2f} "
                    f"L {x1:.2f} {y1 + depth:.2f} A {rx:.2f} {ry:.2f} 0 {f} 0 {x0:.2f} {y0 + depth:.2f} Z")
            walls.append((math.sin(mid), f"<path d='{path}' fill='{dark}'/>"))
        p0 = pt(a0, rx, ry, sx, sy)
        p1 = pt(a1, rx, ry, sx, sy)
        top = (f"M {cx + sx:.2f} {cy + sy:.2f} L {p0[0]:.2f} {p0[1]:.2f} "
               f"A {rx:.2f} {ry:.2f} 0 {flag} 1 {p1[0]:.2f} {p1[1]:.2f} Z")
        lx, ly = pt(mid, rx * 0.63, ry * 0.63, sx, sy)
        pct = 100 * frac
        txt = ""
        if pct >= 5:
            txt = (f"<text x='{lx:.1f}' y='{ly:.1f}' text-anchor='middle' font-size='12' font-weight='700' "
                   f"fill='#0b1017'>{_esc(_MIX_CN.get(label, label))}</text>"
                   f"<text x='{lx:.1f}' y='{ly + 13:.1f}' text-anchor='middle' font-size='11.5' font-weight='700' "
                   f"fill='#0b1017' opacity='.78'>{pct:.1f}%</text>")
        tops.append((math.sin(mid), (
            f"<g class='slice' style='animation-delay:{index * 70}ms'>"
            f"<path d='{top}' fill='{color}' stroke='rgba(6,10,18,.45)' stroke-width='1.4'/>"
            f"<path d='{top}' fill='url(#gloss)' opacity='.35'/>{txt}</g>")))
    walls.sort(key=lambda x: x[0])
    tops.sort(key=lambda x: x[0])
    body = "".join(w for _, w in walls) + "".join(t for _, t in tops)
    return (f"<svg viewBox='0 0 {width} {height}' role='img' aria-label='usage composition'>"
            f"<defs><linearGradient id='gloss' x1='0' y1='0' x2='0' y2='1'>"
            f"<stop offset='0' stop-color='#ffffff' stop-opacity='.55'/>"
            f"<stop offset='1' stop-color='#ffffff' stop-opacity='0'/></linearGradient></defs>{body}</svg>")


def _legend(slices, money=False):
    total = sum(float(v) for _, v, _ in slices) or 1.0
    items = []
    for label, value, color in slices:
        share = 100 * float(value) / total if float(value) else 0.0
        shown = f"\u00a5{_money(value)}" if money else _compact(value)
        name = _MIX_CN.get(label, label)
        items.append(f"<li><i class='sw' style='background:{color}'></i>"
                     f"<span><b>{_esc(name)}</b> {shown} \u00b7 {share:.1f}%</span></li>")
    return "<ul class='legend'>" + "".join(items) + "</ul>"


def _bars(rows):
    if not rows or max(float(v) for _, v, _ in rows) <= 0:
        return "<p class='empty'>no intervals to draw yet</p>"
    peak = max(float(v) for _, v, _ in rows) or 1.0
    body = []
    for i, (label, value, color) in enumerate(rows):
        width = max(1.5, 100 * float(value) / peak)
        body.append("<div class='bar'>"
                    f"<span title='{_esc(label)}'>{_esc(label)}</span>"
                    f"<span class='track'><i class='fill' style='--w:{width:.2f}%;background:{color};"
                    f"animation-delay:{i * 60}ms'></i></span>"
                    f"<span class='num'>{_compact(value)}</span></div>")
    return "<div class='bars'>" + "".join(body) + "</div>"


def _columns(rows):
    if not rows or max(float(v) for _, v, _ in rows) <= 0:
        return "<p class='empty'>no intervals to draw yet</p>"
    peak = max(float(v) for _, v, _ in rows) or 1.0
    cols = []
    for i, (label, value, color) in enumerate(rows):
        height = max(6.0, 112 * float(value) / peak)
        cols.append(f"<div class='col' title='{_esc(label)} \u00b7 {_compact(value)}'>"
                    f"<span class='val'>{_compact(value)}</span>"
                    f"<span class='stem' style='--h:{height:.1f}px;animation-delay:{i * 60}ms;"
                    f"background:linear-gradient(180deg,{_shade(color, 1.25)},{color} 55%,{_shade(color, 0.7)})'></span>"
                    f"<span class='cap'>{_esc(label.split(SEP)[0])}</span></div>")
    return "<div class='cols'>" + "".join(cols) + "</div>"


def _page(title, body):
    return ('<!DOCTYPE html>\n<html lang="zh-CN">\n<head>\n<meta charset="utf-8">\n'
            '<meta http-equiv="refresh" content="300">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
            f'<title>{_esc(title)}</title>\n<style>{_STYLE}</style>\n</head>\n<body>\n'
            f'<div class="wrap">\n{body}\n</div>\n</body>\n</html>\n')


def _head(title, subtitle, picker=""):
    return (f'<header class="bar"><div><h1>{_esc(title)}</h1>'
            f'<p class="sub">{subtitle} \u6bcf 5 \u5206\u949f\u81ea\u52a8\u66f4\u65b0\uff0c'
            f'\u4e5f\u53ef\u4ee5\u9a6c\u4e0a\u5237\u65b0\u3002</p></div>'
            f'<div class="actions">{picker}'
            f'<button class="reload" type="button" onclick="location.reload()">'
            f'\u7acb\u5373\u5237\u65b0</button></div></header>')


def _mix_slices(data):
    return [("input", float(data.get("input_tokens") or 0), _MIX[0][1]),
            ("output", float(data.get("output_tokens") or 0), _MIX[1][1]),
            ("cache", float(data.get("cache_tokens") or 0), _MIX[2][1])]


def _part_slices(parts, limit=8):
    slices = []
    for i, row in enumerate(list(parts)[:limit]):
        slices.append((str(row.get("label") or row.get("name") or "other"),
                       float(row.get("total_tokens") or 0), _PALETTE[i % len(_PALETTE)]))
    rest = sum(float(r.get("total_tokens") or 0) for r in list(parts)[limit:])
    if rest:
        slices.append(("other", rest, "#94a3b8"))
    return slices


def render_task_html(data):
    """The page for one task bill."""
    return render_board_html([data])


def pick_task(tasks, selected, newest_first=True):
    if not tasks:
        return None
    ordered = sorted(tasks, key=lambda item: float(item.get("mtime") or 0), reverse=newest_first)
    if selected:
        if selected.isdigit() and int(selected) < len(ordered):
            return ordered[int(selected)]
        for item in ordered:
            name = f"{item.get('project') or ''} \u00b7 {item.get('task') or ''}"
            if selected == str(item.get("task")) or selected in name:
                return item
    return ordered[0]


def _task_key(item, index):
    return f"{index}:{item.get('project') or ''}:{item.get('task') or ''}"


def _picker(labels, current):
    if len(labels) <= 1:
        return ""
    opts = "".join(
        f"<option value='{_esc(key)}'{' selected' if key == current else ''}>{_esc(text)}</option>"
        for key, text in labels)
    return ("<select onchange=\"location='?task='+encodeURIComponent(this.value)\" "
            f"aria-label='switch task'>{opts}</select>")


def render_board_html(tasks, selected=None):
    ordered = sorted(tasks, key=lambda item: (float(item.get("mtime") or 0),
                                              str(item.get("project") or "")), reverse=True)
    if not ordered:
        body = _head("token usage", "no task bills found yet.") + \
               "<p class='empty'>Start a run and the first API call will create one.</p>"
        return _page("token usage", body)
    keys = [_task_key(item, i) for i, item in enumerate(ordered)]
    chosen = 0
    if selected:
        if selected.isdigit() and int(selected) < len(ordered):
            chosen = int(selected)
        elif selected in keys:
            chosen = keys.index(selected)
    data = ordered[chosen]
    labels = []
    for i, item in enumerate(ordered):
        labels.append((_task_key(item, i),
                       f"{item.get('project') or ''} \u00b7 {item.get('task') or ''} "
                       f"\u00b7 \u00a5{_money((item.get('cost_rmb') or {}).get('total'))}"))
    task = str(data.get("task") or "current task")
    project = str(data.get("project") or "")
    cost = (data.get("cost_rmb") or {}).get("total", "0")
    parts = list(data.get("by_part") or [])
    intervals = list(data.get("by_interval") or [])
    top = data.get("top_part") or (parts[0] if parts else None)
    headline = "no calls yet."
    if top:
        headline = (f"the heaviest part is <b>{_esc(top.get('label') or top.get('name'))}</b> "
                    f"with {int(top.get('total_tokens') or 0):,} tokens "
                    f"({100 * float(top.get('share') or 0):.1f}%)")
    mix = _mix_slices(data)
    part_slices = _part_slices(parts)
    part_rows = [(str(r.get("label") or r.get("name") or ""), float(r.get("total_tokens") or 0),
                  _PALETTE[i % len(_PALETTE)]) for i, r in enumerate(parts)]
    hour_rows = [(str(r.get("label") or ""), float(r.get("total_tokens") or 0), "#4ecdc4")
                 for r in intervals]
    total_tokens = int(data.get("total_tokens") or 0)
    calls = int(data.get("calls") or 0)
    per_call = total_tokens / calls if calls else 0
    cache_share = float(data.get("cache_hit_rate") or 0)
    body = _head(f"{project} \u00b7 {task}" if project else task, headline, _picker(labels, keys[chosen]))
    body += f"""
<section class="stats">
  <div class="stat"><div class="k">settled cost</div><div class="v">\u00a5 {_money(cost)}</div><div class="h">off-peak card</div></div>
  <div class="stat"><div class="k">total tokens</div><div class="v">{_compact(total_tokens)}</div><div class="h">{total_tokens:,}</div></div>
  <div class="stat"><div class="k">calls</div><div class="v">{calls:,}</div><div class="h">about {_compact(int(per_call))} each</div></div>
  <div class="stat"><div class="k">cache hits</div><div class="v">{cache_share:.0f}%</div><div class="h">of prompt tokens</div></div>
</section>
<section class="grid">
  <article class="card">
    <h2>token mix</h2>
    <div class="stage">{_pie3d(mix)}</div>
    {_legend(mix)}
  </article>
  <article class="card">
    <h2>heaviest part</h2>
    {_bars(part_rows)}
  </article>
</section>
<section class="card" style="margin-top:14px">
  <h2>per hour (from the first call)</h2>
  {_columns(hour_rows)}
</section>
<p class="note">reasoning tokens are already inside output. Flash off-peak: input 1 / cache 0.02 / output 4 RMB per
million tokens. Updated {_esc(data.get('updated_at') or '')}.</p>
"""
    return _page(f"{project} \u00b7 {task}", body)
