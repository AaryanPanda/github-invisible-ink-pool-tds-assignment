# Invisible Ink Pool

A shared, de-duplicated pool of labelled training recordings for the TDS **Invisible Ink** bonus.

- Upload training `.jsonl` files. Each recording is fingerprinted by its tilt readings plus its code, so re-uploads and renamed copies are never counted twice.
- Download the pooled `.jsonl`, the predictor script, or a ZIP of everything.
- **Predict online:** drop the six test recordings on the page to get pictures and an auto guess for each. Test files are never stored.
- The model retrains itself in the background after each upload that adds new recordings (a separate process with 1 thread, so the site stays usable).
- `invisible_ink.ipynb` is the same predictor as a Colab notebook.
- Admin delete: open `/?admin` and use the `ADMIN_TOKEN` value from Render's Environment tab.

## Deploy on Render (free)
1. Push this folder to a new GitHub repo.
2. In Render, go to **New → Blueprint**, pick the repo, and approve. `render.yaml` creates the web service and a free Postgres database and wires `DATABASE_URL` in automatically.
3. Open the URL and upload your own file first.

Run locally with SQLite: `pip install -r requirements.txt && uvicorn app:app --reload`
