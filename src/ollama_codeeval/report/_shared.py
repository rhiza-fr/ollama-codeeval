import json
import re
from pathlib import Path

from ollama_codeeval.config import OUTPUT_BASE, OUTPUT_HTML

CANONICAL_TAU = 10.0  # half-credit time (seconds) used throughout all reports


def compute_yield(tasks: list) -> float:
    """Hyperbolic-discounted accuracy at CANONICAL_TAU. Returns value in [0, 1]."""
    N = len(tasks)
    if N == 0:
        return 0.0
    total = 0.0
    for t in tasks:
        elapsed = sum(it.get("total_duration") or 0 for it in t.get("iterations", [])) / 1e9
        if t["final_result"]["exit_code"] == 0:
            total += 1.0 / (1.0 + elapsed / CANONICAL_TAU)
    return total / N

OUTPUT_DIR = OUTPUT_HTML
_STATIC_DIR = Path(__file__).parent


_HUMANEVAL_ALIASES = {"humaneval", "human-eval-enhanced-202307"}

# Display-name overrides: run tag → human-readable label shown in reports
_TAG_DISPLAY_NAMES: dict[str, str] = {
    "cascade2": "cascade",
}


def tag_display_name(d: dict) -> str:
    """Return the display name for a summary record, applying tag overrides."""
    tag = d.get("tag") or ""
    return _TAG_DISPLAY_NAMES.get(tag, d["model"])


def normalize_dataset(name: str) -> str:
    """Normalize dataset name variants to a single canonical form."""
    return "humaneval" if name in _HUMANEVAL_ALIASES else name


def resolve_model_name(name: str, stats: dict | None) -> str:
    """Return the most precise alias for a model name using model_stats.json."""
    resolved = name
    if stats and name in stats:
        digest = stats[name].get("digest")
        if digest:
            aliases = [k for k, v in stats.items() if v.get("digest") == digest]
            if len(aliases) > 1:
                non_latest = [a for a in aliases if not a.endswith(":latest")]
                candidates: list[str] = non_latest if non_latest else aliases
                candidates.sort(key=len)
                resolved = candidates[0]

    if resolved.endswith(":latest") and stats and resolved in stats:
        size = stats[resolved].get("parameter_size", "")
        if size:
            resolved = resolved[: -len(":latest")] + ":" + size.lower()

    return resolved


def load_model_stats() -> dict | None:
    """Load output/model_stats.json, return None if unavailable."""
    path = OUTPUT_BASE / "model_stats.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


HEAD_META = '<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
CSS_BLOCK = "<style>\n" + (_STATIC_DIR / "report.css").read_text(encoding="utf-8") + "\n</style>"
SORT_SCRIPT = "<script>\n" + (_STATIC_DIR / "sort.js").read_text(encoding="utf-8") + "\n</script>"
HLJS_BLOCK = (
    '<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.7.0/styles/default.min.css">\n'
    '<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.7.0/highlight.min.js"></script>\n'
    "<script>document.addEventListener('DOMContentLoaded', () => hljs.highlightAll());</script>"
)


def _nav_html(current: str) -> str:
    """Return sticky top-nav HTML. current is one of: 'index', 'tasks', 'context', 'model'."""
    def _link(href: str, label: str, key: str) -> str:
        cls = ' class="active"' if key == current else ""
        return f'<a href="{href}"{cls}>{label}</a>'

    return (
        '<nav class="top-nav">'
        '<span class="nav-brand">Humaneval LLM Benchmark</span>'
        + _link("index.html", "Overview", "index")
        + _link("tasks.html", "Tasks", "tasks")
        + _link("yield.html", "Yield", "yield")
        + _link("selector.html", "Selector", "selector")
        + _link("cascade.html", "Cascade", "cascade")
        + _link("setup.html", "Setup", "setup")
        + "</nav>"
    )


_DISPLAY_TRUNCATE_LIMIT = 4000

EXPAND_SCRIPT = """<style>
.exp-btn{background:none;border:1px solid #aaa;border-radius:3px;cursor:pointer;font-size:0.75em;padding:1px 6px;margin-left:6px;color:#555}
.exp-btn:hover{background:#eee}
</style>
<script>
function openAnchor(){
  var id=decodeURIComponent(location.hash.slice(1));
  if(!id)return;
  var el=document.getElementById(id);
  if(!el)return;
  var p=el.parentElement;
  while(p){if(p.tagName==='DETAILS')p.open=true;p=p.parentElement;}
  el.scrollIntoView({block:'start'});
}
window.addEventListener('DOMContentLoaded',openAnchor);
window.addEventListener('hashchange',openAnchor);
function expToggle(btn){
  var pre=btn.closest('pre');
  var code=pre.querySelector('code');
  var hint=pre.querySelector('.exp-hint');
  var expanding=btn.dataset.state!=='open';
  if(expanding){
    code.textContent=code.textContent+btn.dataset.more;
    if(window.hljs){delete code.dataset.highlighted;hljs.highlightElement(code);}
    if(hint)hint.hidden=true;
    btn.textContent='Show less';
    btn.dataset.state='open';
  }else{
    code.textContent=btn.dataset.trunc;
    if(window.hljs){delete code.dataset.highlighted;hljs.highlightElement(code);}
    if(hint)hint.hidden=false;
    btn.textContent='Show more';
    btn.dataset.state='closed';
  }
}
</script>"""


def expandable_code_html(s: str, lang: str = "language-python") -> str:
    """Return HTML for a <code> block plus expand controls outside it.
    Caller should wrap in <pre>…</pre> only — do NOT add <code> tags around this."""
    import html as _html
    if len(s) <= _DISPLAY_TRUNCATE_LIMIT:
        return f'<code class="{lang}">{_html.escape(s)}</code>'
    overflow = len(s) - _DISPLAY_TRUNCATE_LIMIT
    trunc_text = s[:_DISPLAY_TRUNCATE_LIMIT]
    rest_text = s[_DISPLAY_TRUNCATE_LIMIT:]
    trunc_attr = _html.escape(trunc_text)   # safe in data-* attribute (escapes &, <, >, ")
    rest_attr = _html.escape(rest_text)
    return (
        f'<code class="{lang}">{trunc_attr}</code>'
        f'<em class="exp-hint" style="color:#888"> … {overflow} more chars</em>'
        f'<button class="exp-btn" data-trunc="{trunc_attr}" data-more="{rest_attr}" data-state="closed" onclick="expToggle(this)">Show more</button>'
    )


def truncate_str(s: str) -> str:
    if len(s) > _DISPLAY_TRUNCATE_LIMIT:
        return s[:_DISPLAY_TRUNCATE_LIMIT] + f" [truncated {len(s) - _DISPLAY_TRUNCATE_LIMIT} characters]"
    return s


def ns_to_seconds(ns):
    if ns is None:
        return "N/A"
    return f"{ns / 1000000000.0:.3f}"


def sanitize_task_id(task_id):
    return task_id.replace("/", "-")


def anchor_id(s: str) -> str:
    """Sanitize a string for use as an HTML id / URL fragment."""
    return re.sub(r"[^A-Za-z0-9/_.-]", "_", s)


def _model_label(record: dict) -> str:
    """Derive display label from iteration-level model names (order-preserving, deduplicated)."""
    seen: dict[str, None] = {}
    for it in record.get("iterations", []):
        m = it.get("model")
        if m:
            seen[m] = None
    return "-".join(seen) if seen else record.get("model", "unknown")
