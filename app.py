"""Invisible Ink Pool - shared, de-duplicated training data for the TDS Invisible Ink bonus."""
import os, io, json, hashlib, zipfile, datetime, collections
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse, JSONResponse
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, ForeignKey, LargeBinary, func
import base64, pickle, time
from typing import List
from sqlalchemy.orm import declarative_base, sessionmaker

DB_URL = os.environ.get("DATABASE_URL", "sqlite:///./inkpool.db")
if DB_URL.startswith("postgres://"):
    DB_URL = DB_URL.replace("postgres://", "postgresql://", 1)
ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN", "")
MAX_BYTES = 15 * 1024 * 1024
ALPHABET = set("ACEFHJKMNPRTUVWXY")
HERE = os.path.dirname(os.path.abspath(__file__))

engine = create_engine(DB_URL, pool_pre_ping=True)
Session = sessionmaker(bind=engine)
Base = declarative_base()

class Upload(Base):
    __tablename__ = "ink_uploads"
    id = Column(Integer, primary_key=True)
    created = Column(DateTime, default=datetime.datetime.utcnow)
    filename = Column(String(300))
    note = Column(String(200))          # optional nickname, never required
    n_records = Column(Integer)
    n_new = Column(Integer)
    raw = Column(Text)                  # original file text

class Recording(Base):
    __tablename__ = "ink_recordings"
    id = Column(Integer, primary_key=True)
    fingerprint = Column(String(64), unique=True, index=True)
    rec_id = Column(String(100))
    code = Column(String(10))
    length = Column(Integer)
    upload_id = Column(Integer, ForeignKey("ink_uploads.id", ondelete="CASCADE"))
    created = Column(DateTime, default=datetime.datetime.utcnow)
    body = Column(Text)

class Model(Base):
    __tablename__ = "ink_models"
    id = Column(Integer, primary_key=True)
    created = Column(DateTime, default=datetime.datetime.utcnow)
    n_recordings = Column(Integer)
    seconds = Column(Integer)
    blob = Column(LargeBinary)

Base.metadata.create_all(engine)

# ---------- model: trained by GitHub Actions (7 GB RAM), uploaded via /api/model ----------
_model_cache = {"id": None, "model": None}

def pool_size():
    with Session() as db:
        return db.query(func.count(Recording.id)).scalar() or 0

def latest_model_meta():
    with Session() as db:
        return db.query(Model.id, Model.created, Model.n_recordings, Model.seconds).order_by(Model.id.desc()).first()

def request_training():
    pass  # training happens on GitHub Actions; see .github/workflows/train.yml

def get_model():
    m = latest_model_meta()
    if not m: return None, None
    if _model_cache["id"] != m.id:
        with Session() as db:
            blob = db.query(Model.blob).filter(Model.id == m.id).scalar()
        _model_cache["model"] = pickle.loads(blob); _model_cache["id"] = m.id
    return _model_cache["model"], m
app = FastAPI(title="Invisible Ink Pool")



def fingerprint(rec):
    """Content fingerprint: code + tilt readings rounded (ignores id, field order, timestamps)."""
    pts = [(round(float(s[1]), 1), round(float(s[2]), 1)) for s in rec["samples"]]
    return hashlib.sha256(json.dumps([rec["code"], pts]).encode()).hexdigest()

def parse(text):
    text = text.strip().lstrip("﻿")
    try:
        o = json.loads(text)
        objs = o if isinstance(o, list) else [o]
    except json.JSONDecodeError:
        objs = []
        for n, line in enumerate(text.splitlines(), 1):
            if line.strip():
                try:
                    objs.append(json.loads(line))
                except json.JSONDecodeError:
                    raise HTTPException(400, f"Line {n} is not valid JSON.")
    good, errors = [], []
    for n, r in enumerate(objs, 1):
        if not isinstance(r, dict):
            errors.append(f"record {n}: not an object"); continue
        code = str(r.get("code", "")).upper()
        s = r.get("samples")
        if not code or not (3 <= len(code) <= 5) or set(code) - ALPHABET:
            errors.append(f"record {n}: missing/invalid 'code' (test files have no answers, upload training files)"); continue
        if not isinstance(s, list) or len(s) < 20 or not all(isinstance(x, list) and len(x) >= 4 for x in s):
            errors.append(f"record {n}: 'samples' must be [t, beta, gamma, pen] rows with pen-down labels"); continue
        r["code"] = code
        good.append(r)
    return good, errors

@app.get("/", response_class=HTMLResponse)
def home():
    return open(os.path.join(HERE, "static", "index.html"), encoding="utf-8").read()

def _model_info():
    m = latest_model_meta()
    return {"trained_on": m.n_recordings if m else 0, "when": m.created.isoformat(timespec="minutes") + "Z" if m else None,
            "seconds": m.seconds if m else None, "pool": pool_size()}

@app.get("/api/stats")
def stats():
    with Session() as db:
        recs = db.query(Recording.code, Recording.length).all()
        ups = db.query(Upload).order_by(Upload.id.desc()).all()
        letters = collections.Counter("".join(c for c, _ in recs))
        return {
            "unique_recordings": len(recs),
            "uploads": len(ups),
            "total_records_uploaded": sum(u.n_records or 0 for u in ups),
            "letters": {k: letters.get(k, 0) for k in sorted(ALPHABET)},
            "lengths": dict(collections.Counter(l for _, l in recs)),
            "model": _model_info(),
            "recent": [{"id": u.id, "when": u.created.isoformat(timespec="minutes") + "Z", "file": u.filename,
                        "note": u.note, "records": u.n_records, "new": u.n_new} for u in ups[:50]],
        }

@app.post("/api/upload")
async def upload(file: UploadFile = File(...), note: str = Form("")):
    raw = await file.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise HTTPException(413, "File too large (max 15 MB).")
    text = raw.decode("utf-8", errors="replace")
    good, errors = parse(text)
    if not good:
        raise HTTPException(400, "No valid labelled recordings. " + "; ".join(errors[:5]))
    with Session() as db:
        up = Upload(filename=(file.filename or "upload")[:300], note=note.strip()[:200], n_records=len(good), n_new=0, raw=text)
        db.add(up); db.flush()
        existing = {f for (f,) in db.query(Recording.fingerprint).all()}
        new = 0
        for r in good:
            fp = fingerprint(r)
            if fp in existing:
                continue
            existing.add(fp); new += 1
            db.add(Recording(fingerprint=fp, rec_id=str(r.get("id", ""))[:100], code=r["code"],
                             length=len(r["code"]), upload_id=up.id, body=json.dumps(r, separators=(",", ":"))))
        up.n_new = new
        db.commit()
        total = db.query(func.count(Recording.id)).scalar()
    if new:
        request_training()
    return {"records": len(good), "new": new, "duplicates": len(good) - new, "skipped": errors[:20], "pool_size": total}

@app.get("/download/pool.jsonl")
def download_pool():
    def gen():
        with Session() as db:
            for (b,) in db.query(Recording.body).order_by(Recording.id).yield_per(200):
                yield b + "\n"
    return StreamingResponse(gen(), media_type="application/x-ndjson",
                             headers={"Content-Disposition": "attachment; filename=air-ink-pool.jsonl"})

@app.get("/download/all.zip")
def download_all():
    buf = io.BytesIO()
    with Session() as db, zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("air-ink-pool.jsonl", "".join(b + "\n" for (b,) in db.query(Recording.body).order_by(Recording.id)))
        for u in db.query(Upload).order_by(Upload.id):
            z.writestr(f"uploads/{u.id:04d}_{os.path.basename(u.filename or 'upload')}", u.raw or "")
        z.write(os.path.join(HERE, "invisible_ink.py"), "invisible_ink.py")
    buf.seek(0)
    return StreamingResponse(buf, media_type="application/zip",
                             headers={"Content-Disposition": "attachment; filename=invisible-ink-pool.zip"})

@app.get("/download/model.pkl")
def download_model():
    with Session() as db:
        blob = db.query(Model.blob).order_by(Model.id.desc()).limit(1).scalar()
    if not blob: raise HTTPException(404, "No model trained yet.")
    return StreamingResponse(io.BytesIO(blob), media_type="application/octet-stream",
                             headers={"Content-Disposition": "attachment; filename=ink_model.pkl"})

@app.get("/download/invisible_ink.py")
def download_script():
    return FileResponse(os.path.join(HERE, "invisible_ink.py"), filename="invisible_ink.py", media_type="text/x-python")

@app.delete("/api/upload/{uid}")
def delete_upload(uid: int, token: str = Query("")):
    if not ADMIN_TOKEN or token != ADMIN_TOKEN:
        raise HTTPException(403, "Bad admin token.")
    with Session() as db:
        db.query(Recording).filter(Recording.upload_id == uid).delete()
        n = db.query(Upload).filter(Upload.id == uid).delete()
        db.commit()
        # re-add recordings that other uploads also contained
        existing = {f for (f,) in db.query(Recording.fingerprint).all()}
        restored = 0
        for u in db.query(Upload).order_by(Upload.id):
            good, _ = parse(u.raw or "")
            for r in good:
                fp = fingerprint(r)
                if fp not in existing:
                    existing.add(fp); restored += 1
                    db.add(Recording(fingerprint=fp, rec_id=str(r.get("id", ""))[:100], code=r["code"],
                                     length=len(r["code"]), upload_id=u.id, body=json.dumps(r, separators=(",", ":"))))
        db.commit()
    request_training()
    return {"deleted": n, "restored_from_other_uploads": restored}

@app.post("/api/predict")
def predict(files: List[UploadFile] = File(...)):
    """Test recordings are analysed in memory and never stored."""
    import invisible_ink as ink
    M, meta = get_model()
    if M is None:
        raise HTTPException(503, "No model yet. Upload training data and wait for training to finish.")
    recs = []
    for f in files[:12]:
        raw = f.file.read(MAX_BYTES + 1)
        if len(raw) > MAX_BYTES: raise HTTPException(413, f"{f.filename} too large")
        try:
            recs += ink.parse_text(raw.decode("utf-8", errors="replace"), f.filename or "")
        except Exception:
            raise HTTPException(400, f"{f.filename}: could not read JSON/JSONL")
    if not recs:
        raise HTTPException(400, "No recordings with 'samples' found.")
    out = []
    for n, l in enumerate(recs[:12], 1):
        r = ink.analyze(M, l, n)
        out.append({"n": n, "file": r["file"], "id": r["id"], "length": r["length"], "guess": r["guess"],
                    "alts": r["alts"], "png": base64.b64encode(r["png"]).decode()})
    return {"model_trained_on": meta.n_recordings, "results": out}

@app.post("/api/model")
def upload_model(file: UploadFile = File(...), n_recordings: int = Form(...), seconds: int = Form(0), token: str = Query("")):
    """Admin only (called by the GitHub Action). The pickle is only ever loaded from this trusted source."""
    if not ADMIN_TOKEN or token != ADMIN_TOKEN:
        raise HTTPException(403, "Bad admin token.")
    blob = file.file.read(200 * 1024 * 1024)
    try:
        import invisible_ink as ink
        M = pickle.loads(blob)
        with Session() as db:
            body = db.query(Recording.body).first()
        if body:
            r = ink.analyze(M, ink.norm_rec(json.loads(body[0])), 1)   # smoke test before accepting
            assert len(r["guess"]) >= 3
    except Exception as e:
        raise HTTPException(400, f"Model failed to load/run: {e}")
    with Session() as db:
        db.add(Model(n_recordings=n_recordings, seconds=seconds, blob=blob)); db.commit()
        old = [i for (i,) in db.query(Model.id).order_by(Model.id.desc()).offset(2).all()]
        if old:
            db.query(Model).filter(Model.id.in_(old)).delete(synchronize_session=False); db.commit()
    _model_cache["id"] = None; _model_cache["model"] = None
    return {"ok": True, "n_recordings": n_recordings}


@app.get("/healthz")
def health():
    return {"ok": True}
