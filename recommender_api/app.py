from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle, pandas as pd, numpy as np

app = FastAPI()

# ── Load artefacts on startup ──────────────────────────────────
with open('model_artefacts/cf_model.pkl', 'rb') as f:
    cf_model = pickle.load(f)
with open('model_artefacts/meta_clf.pkl', 'rb') as f:
    meta_clf = pickle.load(f)
with open('model_artefacts/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('model_artefacts/cos_sim_matrix.pkl', 'rb') as f:
    cos_sim_matrix = pickle.load(f)
with open('model_artefacts/product_idx.pkl', 'rb') as f:
    product_idx = pickle.load(f)
with open('model_artefacts/new_items.pkl', 'rb') as f:
    new_items = pickle.load(f)
with open('model_artefacts/trending_items.pkl', 'rb') as f:
    trending_items = pickle.load(f)

product_meta  = pd.read_parquet('model_artefacts/product_meta.parquet')
customer_df   = pd.read_parquet('model_artefacts/customer_df.parquet')
train_pairs   = pd.read_parquet('model_artefacts/train_pairs.parquet')
trend_df      = pd.read_parquet('model_artefacts/trend_df.parquet')

COLD_START_THRESHOLD = 5

class RecommendRequest(BaseModel):
    customer_id: str
    top_k: int = 10

@app.post("/recommend")
def recommend(req: RecommendRequest):
    try:
        recs, model_used, reason = get_hybrid_recommendation(
            req.customer_id, top_k=req.top_k
        )
        return {
            "customer_id"  : req.customer_id,
            "model_used"   : model_used,
            "reason"       : reason,
            "recommendations": recs[
                ['StockCode', 'Description', 'AvgPrice', 'item_type']
            ].to_dict(orient='records')
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "ok"}