# FytMe

Short description of the project and how it works under the hood.

## What it does
- Outfit recommendations from a curated clothing catalog with style and color awareness.
- Multi-attribute labeling of images (category, color, top styles) using CLIP zero-shot inference.
- Pairwise scoring of outfits plus diversity sampling to avoid repetitive looks.
- Debug endpoints for reindexing the catalog and inspecting recommendations.

## Tech stack
- Backend: FastAPI, Python, CLIP ViT-B/32 (CPU/GPU), Pillow, Torch.
- Frontend: React (Vite/esbuild), fetches API for outfits and image assets.
- Data: Image assets in `backend/assets/clothes`, cached metadata in `backend/assets/catalog.json`.

## How it works
- `app/image_indexer.py` loads CLIP once, scans the assets folder, infers category/color/styles, renames files to clean slugs with a hash, saves embeddings and labels to the catalog cache.
- `app/recommender.py` loads the catalog, blends item affinity, pairwise compatibility, and style-aware scoring, then adds lightweight diversity on color/style to keep results varied.
- `app/api.py` exposes routes for outfits, images, and `/api/debug/reindex` to force a fresh index.

## Run locally
- Backend: `cd backend && .\start.ps1` (PowerShell) then open http://localhost:8000/docs.
- Reindex: `Invoke-RestMethod -Uri "http://localhost:8000/api/debug/reindex" -Method POST` after adding/removing images.
- Frontend: `cd frontend && npm install && npm run dev` (or `npm run build` for production).

## Notes
- Set `INDEXER_DEBUG=1` before starting the backend to log detailed label scores.
- Cached catalog avoids re-encoding unchanged images; hash changes force refresh even with same filenames.
