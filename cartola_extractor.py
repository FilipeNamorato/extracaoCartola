"""
cartola_extractor.py
--------------------
Extrai dados da API do Cartola FC, enriquece com colunas calculadas
e salva em data/. Roda via GitHub Actions diariamente.

Colunas extras geradas em atletas_enriquecido.csv:
  - mandante      : True se o clube joga em casa nessa rodada
  - adversario    : abreviação do adversário na rodada
  - tendencia     : 'alta' | 'baixa' | 'estavel' com base na variação
  - custo_beneficio : media / preco
  - cb_rank       : posição no ranking de custo-benefício dentro da posição
  - armadilha     : True se preço acima da mediana mas média abaixo da mediana da posição
  - status_label  : texto legível do status (Provável, Dúvida, etc.)
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

STATUS_LABEL = {
    2: "Dúvida",
    3: "Suspenso",
    5: "Contundido",
    6: "Nulo",
    7: "Provável",
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
# ENRIQUECIMENTO
# ─────────────────────────────────────────────────────────────

def enriquecer(df_mercado: pd.DataFrame, df_partidas: pd.DataFrame) -> pd.DataFrame:
    df = df_mercado.copy()

    # ── Mandante / adversário ────────────────────────────────
    mapa_confronto = {}

    if not df_partidas.empty:
        col_casa     = next((c for c in df_partidas.columns if "casa_id" in c), None)
        col_vis      = next((c for c in df_partidas.columns if "visitante_id" in c), None)
        col_casa_abr = next((c for c in df_partidas.columns if "casa_abreviacao" in c or "casa.abreviacao" in c), None)
        col_vis_abr  = next((c for c in df_partidas.columns if "visitante_abreviacao" in c or "visitante.abreviacao" in c), None)

        for _, p in df_partidas.iterrows():
            try:
                id_casa  = int(p[col_casa]) if col_casa else None
                id_vis   = int(p[col_vis])  if col_vis  else None
                abr_casa = str(p[col_casa_abr]) if col_casa_abr else str(id_casa)
                abr_vis  = str(p[col_vis_abr])  if col_vis_abr  else str(id_vis)

                if id_casa:
                    mapa_confronto[id_casa] = {"mandante": True,  "adversario": abr_vis}
                if id_vis:
                    mapa_confronto[id_vis]  = {"mandante": False, "adversario": abr_casa}
            except Exception:
                continue

    def get_confronto(clube_id, campo, default):
        try:
            return mapa_confronto.get(int(clube_id), {}).get(campo, default)
        except Exception:
            return default

    df["mandante"]   = df["clube_id"].apply(lambda x: get_confronto(x, "mandante", None))
    df["adversario"] = df["clube_id"].apply(lambda x: get_confronto(x, "adversario", "—"))

    # ── Tendência ────────────────────────────────────────────
    def calcular_tendencia(v):
        try:
            v = float(v)
            if v > 0.5:  return "alta"
            if v < -0.5: return "baixa"
            return "estavel"
        except Exception:
            return "estavel"

    df["tendencia"] = df["variacao"].apply(calcular_tendencia)

    # ── Custo-benefício e rank por posição ───────────────────
    df["preco"] = pd.to_numeric(df["preco"], errors="coerce").fillna(0)
    df["media"] = pd.to_numeric(df["media"], errors="coerce").fillna(0)

    df["custo_beneficio"] = df.apply(
        lambda r: round(r["media"] / r["preco"], 3) if r["preco"] > 0 else 0, axis=1
    )

    df["cb_rank"] = (
        df[df["preco"] > 0]
        .groupby("posicao")["custo_beneficio"]
        .rank(ascending=False, method="min")
        .reindex(df.index)
        .fillna(0)
        .astype(int)
    )

    # ── Armadilha ────────────────────────────────────────────
    mediana_preco = df.groupby("posicao")["preco"].transform("median")
    mediana_media = df.groupby("posicao")["media"].transform("median")
    df["armadilha"] = (df["preco"] > mediana_preco) & (df["media"] < mediana_media)

    # ── Status legível ───────────────────────────────────────
    df["status_label"] = df["status_id"].apply(
        lambda x: STATUS_LABEL.get(int(x), "Desconhecido") if pd.notna(x) else "Desconhecido"
    )

    return df


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
dados_brutos = {}

for key, (normalizador, nome_arquivo) in EXTRATORES.items():
    print(f"Extraindo {key}...")
    try:
        raw = get_json(key)

        with open(DATA_DIR / f"{nome_arquivo}.json", "w", encoding="utf-8") as f:
            json.dump(raw, f, ensure_ascii=False, indent=2)

        df = normalizador(raw)
        df.to_csv(DATA_DIR / f"{nome_arquivo}.csv", index=False, encoding="utf-8-sig")
        dados_brutos[key] = df

        print(f"  OK — {len(df)} registros")
        log.append({"endpoint": key, "registros": len(df), "status": "OK", "erro": ""})

    except Exception as e:
        print(f"  ERRO: {e}")
        log.append({"endpoint": key, "registros": 0, "status": "ERRO", "erro": str(e)})
        dados_brutos[key] = pd.DataFrame()

# ── CSV enriquecido ──────────────────────────────────────────
print("Gerando CSV enriquecido...")
try:
    df_mercado  = dados_brutos.get("mercado", pd.DataFrame())
    df_partidas = dados_brutos.get("partidas", pd.DataFrame())

    if not df_mercado.empty:
        df_enriquecido = enriquecer(df_mercado, df_partidas)
        df_enriquecido.to_csv(DATA_DIR / "atletas_enriquecido.csv", index=False, encoding="utf-8-sig")
        print(f"  OK — {len(df_enriquecido)} atletas enriquecidos")
        log.append({"endpoint": "enriquecido", "registros": len(df_enriquecido), "status": "OK", "erro": ""})
    else:
        print("  SKIP — mercado vazio")

except Exception as e:
    print(f"  ERRO no enriquecimento: {e}")
    log.append({"endpoint": "enriquecido", "registros": 0, "status": "ERRO", "erro": str(e)})

# ── Log ──────────────────────────────────────────────────────
ts = datetime.now().strftime("%Y-%m-%d %H:%M")
log_df = pd.DataFrame(log)
log_df.insert(0, "timestamp", ts)

log_path = DATA_DIR / "log.csv"
if log_path.exists():
    log_existente = pd.read_csv(log_path)
    log_df = pd.concat([log_existente, log_df], ignore_index=True)

log_df.to_csv(log_path, index=False, encoding="utf-8-sig")
print(f"\nExtração concluída — {ts}")