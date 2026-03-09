"""
cartola_extractor.py
--------------------
Extrai dados brutos da API do Cartola FC e salva em data/.
Roda via GitHub Actions diariamente.
"""

import json
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime

BASE_URL = "https://api.cartola.globo.com"

ENDPOINTS = {
    "mercado":   "/atletas/mercado",
    "pontuados": "/atletas/pontuados",
    "partidas":  "/partidas",
    "rodadas":   "/rodadas",
}

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json",
}

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────
# REQUISIÇÃO
# ─────────────────────────────────────────────────────────────

def get_json(endpoint_key: str) -> dict:
    url = BASE_URL + ENDPOINTS[endpoint_key]
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    return resp.json()


# ─────────────────────────────────────────────────────────────
# NORMALIZADORES
# ─────────────────────────────────────────────────────────────

def normalizar_mercado(raw: dict) -> pd.DataFrame:
    atletas  = raw.get("atletas", [])
    clubes   = {int(k): v.get("abreviacao", k) for k, v in raw.get("clubes", {}).items()}
    posicoes = {int(k): v.get("nome", k)        for k, v in raw.get("posicoes", {}).items()}

    rows = []
    for a in atletas:
        scouts = a.get("scout") or {}
        rows.append({
            "atleta_id":     a.get("atleta_id"),
            "nome":          a.get("apelido", a.get("nome")),
            "clube_id":      a.get("clube_id"),
            "clube":         clubes.get(a.get("clube_id"), a.get("clube_id")),
            "posicao_id":    a.get("posicao_id"),
            "posicao":       posicoes.get(a.get("posicao_id"), a.get("posicao_id")),
            "status_id":     a.get("status_id"),
            "preco":         a.get("preco_num"),
            "variacao":      a.get("variacao_num"),
            "media":         a.get("media_num"),
            "jogos":         a.get("jogos_num"),
            "pontos_rodada": a.get("pontos_num"),
            **{f"scout_{k}": v for k, v in scouts.items()},
        })
    return pd.DataFrame(rows)


def normalizar_pontuados(raw: dict) -> pd.DataFrame:
    atletas = raw.get("atletas", {})
    clubes  = {int(k): v.get("abreviacao", k) for k, v in raw.get("clubes", {}).items()}

    rows = []
    for atleta_id, a in atletas.items():
        scouts = a.get("scout") or {}
        rows.append({
            "atleta_id":  int(atleta_id),
            "nome":       a.get("apelido", a.get("nome")),
            "clube_id":   a.get("clube_id"),
            "clube":      clubes.get(a.get("clube_id"), a.get("clube_id")),
            "posicao_id": a.get("posicao_id"),
            "pontos":     a.get("pontos_num"),
            "preco":      a.get("preco_num"),
            "variacao":   a.get("variacao_num"),
            **{f"scout_{k}": v for k, v in scouts.items()},
        })
    return pd.DataFrame(rows)


def normalizar_partidas(raw: dict) -> pd.DataFrame:
    lista = raw.get("partidas", raw) if isinstance(raw, dict) else raw
    if isinstance(lista, dict):
        lista = list(lista.values())
    return pd.json_normalize(lista) if lista else pd.DataFrame()


def normalizar_rodadas(raw: list) -> pd.DataFrame:
    return pd.json_normalize(raw) if raw else pd.DataFrame()


# ─────────────────────────────────────────────────────────────
# EXECUÇÃO
# ─────────────────────────────────────────────────────────────

EXTRATORES = {
    "mercado":   (normalizar_mercado,   "atletas_mercado"),
    "pontuados": (normalizar_pontuados, "atletas_pontuados"),
    "partidas":  (normalizar_partidas,  "partidas"),
    "rodadas":   (normalizar_rodadas,   "rodadas"),
}

log = []

for key, (normalizador, nome_arquivo) in EXTRATORES.items():
    print(f"Extraindo {key}...")
    try:
        raw = get_json(key)

        # JSON bruto
        with open(DATA_DIR / f"{nome_arquivo}.json", "w", encoding="utf-8") as f:
            json.dump(raw, f, ensure_ascii=False, indent=2)

        # CSV normalizado
        df = normalizador(raw)
        df.to_csv(DATA_DIR / f"{nome_arquivo}.csv", index=False, encoding="utf-8-sig")

        print(f"  OK — {len(df)} registros")
        log.append({"endpoint": key, "registros": len(df), "status": "OK", "erro": ""})

    except Exception as e:
        print(f"  ERRO: {e}")
        log.append({"endpoint": key, "registros": 0, "status": "ERRO", "erro": str(e)})

# Salva log com timestamp
ts = datetime.now().strftime("%Y-%m-%d %H:%M")
log_df = pd.DataFrame(log)
log_df.insert(0, "timestamp", ts)

log_path = DATA_DIR / "log.csv"
if log_path.exists():
    log_existente = pd.read_csv(log_path)
    log_df = pd.concat([log_existente, log_df], ignore_index=True)

log_df.to_csv(log_path, index=False, encoding="utf-8-sig")
print(f"\nExtração concluída — {ts}")
