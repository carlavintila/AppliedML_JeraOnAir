from __future__ import annotations

import json
import mimetypes
import socket
import sys
import traceback
import webbrowser
from email.parser import BytesParser
from email.policy import default
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path

from dataset_converter import convert_files, read_table


HOST = "127.0.0.1"
DEFAULT_PORT = 8765
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Jera on Air - Data Converter</title>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#07091a;--panel:#111827;--panel2:#162033;--line:#24334f;--text:#e7edf8;--muted:#92a1bb;
  --blue:#3b82f6;--green:#10b981;--pink:#ec4899;--amber:#f59e0b;--red:#ef4444;
}
body{min-height:100vh;background:linear-gradient(180deg,#07091a 0%,#0b1020 48%,#080a16 100%);color:var(--text);font-family:Inter,Segoe UI,Arial,sans-serif;font-size:15px;line-height:1.5}
.wrap{max-width:1180px;margin:0 auto;padding:42px 22px 56px}
.top{display:flex;justify-content:space-between;gap:18px;align-items:flex-end;margin-bottom:26px}
.eyebrow{color:#7dd3fc;text-transform:uppercase;letter-spacing:.13em;font-size:11px;font-weight:800;margin-bottom:8px}
h1{font-size:clamp(28px,5vw,48px);line-height:1.02;font-weight:900}
.sub{color:var(--muted);max-width:680px;margin-top:10px}
.badge{border:1px solid var(--line);background:#0d1426;border-radius:999px;padding:8px 12px;color:#bad0f4;font-weight:700;font-size:12px;white-space:nowrap}
.grid{display:grid;grid-template-columns:1.05fr .95fr;gap:18px;align-items:start}
@media(max-width:920px){.grid{grid-template-columns:1fr}.top{display:block}.badge{display:inline-block;margin-top:16px}}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:8px;box-shadow:0 18px 40px rgba(0,0,0,.25)}
.panel-h{padding:18px 20px;border-bottom:1px solid var(--line);display:flex;align-items:center;gap:10px}
.num{width:26px;height:26px;border-radius:50%;display:grid;place-items:center;background:var(--blue);font-size:12px;font-weight:900}
.panel-h h2{font-size:16px}
.panel-b{padding:20px}
.field{margin-bottom:16px}
label{display:block;color:#c9d7ee;font-size:12px;font-weight:800;letter-spacing:.04em;text-transform:uppercase;margin-bottom:7px}
input[type="number"],input[type="date"],input[type="file"]{width:100%;background:#0b1222;color:var(--text);border:1px solid var(--line);border-radius:6px;padding:11px 12px;font:inherit}
input[type="file"]{padding:10px}
.row{display:grid;grid-template-columns:1fr 1fr;gap:14px}
@media(max-width:620px){.row{grid-template-columns:1fr}}
.hint{color:var(--muted);font-size:13px;margin-top:7px}
.actions{display:flex;gap:10px;align-items:center;margin-top:18px;flex-wrap:wrap}
button,.download{border:0;border-radius:6px;background:var(--green);color:#03140e;font-weight:900;padding:12px 16px;cursor:pointer;text-decoration:none;font:inherit}
button.secondary{background:#1c2940;color:#dbe8ff;border:1px solid var(--line)}
button:disabled{opacity:.55;cursor:not-allowed}
.download{display:none;background:var(--amber);color:#1b1101}
.cards{display:grid;grid-template-columns:repeat(2,1fr);gap:12px;margin-bottom:16px}
.card{background:#0b1222;border:1px solid var(--line);border-radius:8px;padding:16px}
.card .k{color:var(--muted);font-size:11px;font-weight:800;text-transform:uppercase;letter-spacing:.08em}
.card .v{font-size:28px;font-weight:900;margin-top:6px}
.status{border:1px solid var(--line);background:#0b1222;border-radius:8px;padding:14px;color:var(--muted);min-height:52px}
.status.ok{border-color:rgba(16,185,129,.6);color:#b7f7dd}
.status.err{border-color:rgba(239,68,68,.7);color:#fecaca}
.warn{margin-top:12px;color:#fcd34d;font-size:13px}
table{width:100%;border-collapse:collapse;margin-top:16px;font-size:12px}
th,td{border-bottom:1px solid #1b2941;padding:8px 6px;text-align:left;white-space:nowrap}
th{color:#93c5fd;font-weight:800}
.table-wrap{overflow:auto;max-height:330px}
.steps{display:grid;gap:10px}
.step{display:grid;grid-template-columns:28px 1fr;gap:10px;align-items:start;color:#c4d0e4}
.dot{width:22px;height:22px;border-radius:50%;display:grid;place-items:center;border:1px solid var(--line);color:#93c5fd;font-size:12px;font-weight:900}
.small{font-size:13px;color:var(--muted)}
</style>
</head>
<body>
<div class="wrap">
  <div class="top">
    <div>
      <div class="eyebrow">Prediction-ready data pipeline</div>
      <h1>Jera on Air Data Converter</h1>
      <p class="sub">Upload raw ticket sales and new Instagram/Facebook marketing exports. The app builds the same column format as <b>final.csv</b>, including sales lags, rolling marketing features, ticket-type features, and model-ready validation.</p>
    </div>
    <div class="badge">Output: final_for_prediction.csv</div>
  </div>

  <div class="grid">
    <form id="convertForm" class="panel">
      <div class="panel-h"><div class="num">1</div><h2>Upload data</h2></div>
      <div class="panel-b">
        <div class="row">
          <div class="field">
            <label for="festivalYear">Festival year</label>
            <input id="festivalYear" name="festival_year" type="number" min="2022" max="2100" value="2027" required>
          </div>
          <div class="field">
            <label for="festivalDate">Festival start date</label>
            <input id="festivalDate" name="festival_date" type="date" value="2027-06-24" required>
          </div>
        </div>
        <div class="field">
          <label for="ambassadorExpires">Ambassador expiry date</label>
          <input id="ambassadorExpires" name="ambassador_expires" type="date">
          <div class="hint">Optional. If left empty, the converter estimates a 30-day ambassador window from the first ambassador sale.</div>
        </div>
        <div class="field">
          <label for="ticketFile">Ticket sale file</label>
          <input id="ticketFile" name="ticket_file" type="file" accept=".csv,.xlsx,.xls" required>
          <div class="hint">Works with raw files like “Ticket sale data 2026.xlsx” or standardized ticket templates.</div>
        </div>
        <div class="field">
          <label for="marketingFiles">Marketing files</label>
          <input id="marketingFiles" name="marketing_files" type="file" accept=".csv,.xlsx,.xls" multiple>
          <div class="hint">Add Instagram exports, Facebook exports, or daily feature files. Multiple files are merged by date.</div>
        </div>
        <div class="actions">
          <button id="runBtn" type="submit">Convert dataset</button>
          <button class="secondary" type="reset">Clear</button>
          <a id="downloadLink" class="download" download="final_for_prediction.csv">Download CSV</a>
        </div>
      </div>
    </form>

    <div class="panel">
      <div class="panel-h"><div class="num">2</div><h2>Pipeline checks</h2></div>
      <div class="panel-b">
        <div class="steps">
          <div class="step"><div class="dot">A</div><div><b>Tickets</b><div class="small">Transactions are grouped per sale date; ticket names become Early Bird, Full Weekend, Single Day, Ambassador, Camping, or Other.</div></div></div>
          <div class="step"><div class="dot">B</div><div><b>Marketing</b><div class="small">Instagram and Facebook exports are aggregated into daily reach, engagement, views, post counts, lags, and rolling windows.</div></div></div>
          <div class="step"><div class="dot">C</div><div><b>Prediction format</b><div class="small">The final file is ordered exactly like the project’s final.csv and includes every feature used by testing.py.</div></div></div>
        </div>
        <div id="status" class="status" style="margin-top:18px">Ready for files.</div>
      </div>
    </div>
  </div>

  <div class="panel" style="margin-top:18px">
    <div class="panel-h"><div class="num">3</div><h2>Result preview</h2></div>
    <div class="panel-b">
      <div class="cards">
        <div class="card"><div class="k">Rows</div><div class="v" id="rows">-</div></div>
        <div class="card"><div class="k">Tickets</div><div class="v" id="tickets">-</div></div>
        <div class="card"><div class="k">Date range</div><div class="v" id="range" style="font-size:18px">-</div></div>
        <div class="card"><div class="k">Marketing days</div><div class="v" id="marketing">-</div></div>
      </div>
      <div id="warnings" class="warn"></div>
      <div class="table-wrap"><table id="preview"></table></div>
    </div>
  </div>
</div>

<script>
const form = document.getElementById('convertForm');
const statusBox = document.getElementById('status');
const runBtn = document.getElementById('runBtn');
const downloadLink = document.getElementById('downloadLink');
const preview = document.getElementById('preview');
const warnings = document.getElementById('warnings');

function setStatus(text, cls='') {
  statusBox.className = 'status ' + cls;
  statusBox.textContent = text;
}
function fmt(n){ return Number(n || 0).toLocaleString(); }
function renderPreview(rows) {
  preview.innerHTML = '';
  if (!rows || !rows.length) return;
  const cols = Object.keys(rows[0]);
  const thead = document.createElement('thead');
  const trh = document.createElement('tr');
  cols.forEach(c => { const th = document.createElement('th'); th.textContent = c; trh.appendChild(th); });
  thead.appendChild(trh);
  const tbody = document.createElement('tbody');
  rows.forEach(r => {
    const tr = document.createElement('tr');
    cols.forEach(c => { const td = document.createElement('td'); td.textContent = r[c]; tr.appendChild(td); });
    tbody.appendChild(tr);
  });
  preview.appendChild(thead);
  preview.appendChild(tbody);
}

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  runBtn.disabled = true;
  downloadLink.style.display = 'none';
  warnings.textContent = '';
  setStatus('Converting files...');
  try {
    const response = await fetch('/convert', { method: 'POST', body: new FormData(form) });
    const payload = await response.json();
    if (!response.ok || !payload.ok) throw new Error(payload.error || 'Conversion failed.');
    const blob = new Blob([payload.csv], { type: 'text/csv;charset=utf-8' });
    downloadLink.href = URL.createObjectURL(blob);
    downloadLink.style.display = 'inline-block';
    document.getElementById('rows').textContent = fmt(payload.summary.rows);
    document.getElementById('tickets').textContent = fmt(payload.summary.total_tickets);
    document.getElementById('range').textContent = payload.summary.start_date + ' to ' + payload.summary.end_date;
    document.getElementById('marketing').textContent = fmt(payload.summary.marketing_days);
    renderPreview(payload.preview);
    if (payload.summary.warnings.length) warnings.textContent = payload.summary.warnings.join(' ');
    setStatus('Conversion complete. The CSV is ready to download.', 'ok');
  } catch (err) {
    setStatus(err.message, 'err');
  } finally {
    runBtn.disabled = false;
  }
});
</script>
</body>
</html>
"""


def parse_multipart(headers: dict[str, str], body: bytes) -> dict[str, list[dict[str, object]]]:
    content_type = headers.get("content-type", "")
    message_bytes = (
        f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode("utf-8") + body
    )
    message = BytesParser(policy=default).parsebytes(message_bytes)
    fields: dict[str, list[dict[str, object]]] = {}
    for part in message.iter_parts():
        disposition = part.get_content_disposition()
        if disposition != "form-data":
            continue
        name = part.get_param("name", header="content-disposition")
        filename = part.get_filename()
        payload = part.get_payload(decode=True) or b""
        fields.setdefault(name, []).append({"filename": filename, "value": payload})
    return fields


def first_text(fields: dict[str, list[dict[str, object]]], name: str, default_value: str = "") -> str:
    if name not in fields or not fields[name]:
        return default_value
    return bytes(fields[name][0]["value"]).decode("utf-8").strip()


def file_items(fields: dict[str, list[dict[str, object]]], name: str) -> list[tuple[str, bytes]]:
    items = []
    for item in fields.get(name, []):
        filename = str(item.get("filename") or "")
        data = bytes(item["value"])
        if filename and data:
            items.append((filename, data))
    return items


class ConverterHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: object) -> None:
        sys.stdout.write("%s - %s\n" % (self.address_string(), fmt % args))

    def send_json(self, status: int, payload: dict[str, object]) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        if self.path not in {"/", "/index.html"}:
            self.send_error(404)
            return
        data = HTML.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self) -> None:
        if self.path != "/convert":
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            fields = parse_multipart({k.lower(): v for k, v in self.headers.items()}, self.rfile.read(length))
            ticket_uploads = file_items(fields, "ticket_file")
            if not ticket_uploads:
                raise ValueError("Please upload a ticket sale file.")
            festival_year = int(first_text(fields, "festival_year"))
            festival_date = first_text(fields, "festival_date")
            ambassador_expires = first_text(fields, "ambassador_expires") or None

            ticket_name, ticket_bytes = ticket_uploads[0]
            ticket_df = read_table(BytesIO(ticket_bytes), ticket_name)
            marketing = []
            for filename, data in file_items(fields, "marketing_files"):
                marketing.append((filename, read_table(BytesIO(data), filename)))

            final_df, summary = convert_files(
                ticket_file=(ticket_name, ticket_df),
                marketing_files=marketing,
                festival_year=festival_year,
                festival_date=festival_date,
                ambassador_expires=ambassador_expires,
            )
            out_path = OUTPUT_DIR / "final_for_prediction.csv"
            final_df.to_csv(out_path, index=False)
            csv_text = final_df.to_csv(index=False)
            preview_cols = [
                "sale_date", "festival_year", "tickets_sold", "sales_lag_1",
                "sales_roll_7_prior", "marketing_total_engagement_roll_7_prior",
                "Full Weekend_roll_3d", "days_until_ambassador_expires",
            ]
            preview_cols = [c for c in preview_cols if c in final_df.columns]
            payload = {
                "ok": True,
                "summary": {
                    "rows": summary.rows,
                    "start_date": summary.start_date,
                    "end_date": summary.end_date,
                    "total_tickets": summary.total_tickets,
                    "marketing_days": summary.marketing_days,
                    "missing_model_features": summary.missing_model_features,
                    "warnings": summary.warnings,
                    "saved_to": str(out_path),
                },
                "preview": final_df[preview_cols].head(20).to_dict(orient="records"),
                "csv": csv_text,
            }
            self.send_json(200, payload)
        except Exception as exc:
            traceback.print_exc()
            self.send_json(400, {"ok": False, "error": str(exc)})


def pick_port(start: int = DEFAULT_PORT) -> int:
    for port in range(start, start + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind((HOST, port))
                return port
            except OSError:
                continue
    raise RuntimeError("No free local port found.")


def main() -> None:
    port = pick_port()
    url = f"http://{HOST}:{port}"
    server = ThreadingHTTPServer((HOST, port), ConverterHandler)
    (OUTPUT_DIR / "converter_app_url.txt").write_text(url, encoding="utf-8")
    print(f"Jera data converter running at {url}")
    if "--no-browser" not in sys.argv:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    try:
        server.serve_forever()
    except Exception:
        (OUTPUT_DIR / "converter_app_error.log").write_text(traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
