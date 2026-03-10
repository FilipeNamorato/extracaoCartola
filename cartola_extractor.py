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
import os
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone, timedelta

BASE_URL = "https://api.cartola.globo.com"

ENDPOINTS = {
    "mercado":   "/atletas/mercado",
    "pontuados": "/atletas/pontuados",
    "partidas":  "/partidas",
    "rodadas":   "/rodadas",
    "status":    "/mercado/status",
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

ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "")
ODDS_URL     = "https://api.the-odds-api.com/v4/sports/soccer_brazil_campeonato/odds"

# Mapa de nomes da Odds API → abreviações do Cartola
NOMES_PARA_ABR = {
    "Flamengo":            "FLA",
    "Palmeiras":           "PAL",
    "Atletico Mineiro":    "CAM",
    "Fluminense":          "FLU",
    "Corinthians":         "COR",
    "Sao Paulo":           "SAO",
    "Internacional":       "INT",
    "Gremio":              "GRE",
    "Grêmio":              "GRE",
    "Botafogo":            "BOT",
    "Vasco da Gama":       "VAS",
    "Bahia":               "BAH",
    "Cruzeiro":            "CRU",
    "Atletico Paranaense": "CAP",
    "Santos":              "SAN",
    "Vitoria":             "VIT",
    "Bragantino-SP":       "RBB",
    "Mirassol":            "MIR",
    "Chapecoense":         "CHA",
    "Coritiba":            "CFC",
    "Remo":                "REM",
}

BRT = timezone(timedelta(hours=-3))

DATA_DIR = Path("docs/data")
DATA_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────
# REQUISIÇÃO
# ─────────────────────────────────────────────────────────────

def get_json(endpoint_key: str) -> dict:
    url = BASE_URL + ENDPOINTS[endpoint_key]
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    return resp.json()


# ─────────────────────────────────────────────────────────────
# ODDS API
# ─────────────────────────────────────────────────────────────

def get_odds() -> pd.DataFrame:
    """Busca odds do Brasileirão e retorna DataFrame com colunas por abreviação."""
    if not ODDS_API_KEY:
        print("  SKIP — ODDS_API_KEY não definida")
        return pd.DataFrame()

    resp = requests.get(ODDS_URL, params={
        "apiKey":      ODDS_API_KEY,
        "regions":     "eu",
        "markets":     "h2h",
        "oddsFormat":  "decimal",
    }, timeout=15)
    resp.raise_for_status()
    jogos = resp.json()

    rows = []
    for jogo in jogos:
        abr_casa = NOMES_PARA_ABR.get(jogo["home_team"])
        abr_vis  = NOMES_PARA_ABR.get(jogo["away_team"])
        if not abr_casa or not abr_vis:
            continue

        odd_casa = odd_vis = odd_empate = None
        for bm in jogo.get("bookmakers", [])[:1]:
            for market in bm.get("markets", []):
                if market["key"] == "h2h":
                    for o in market["outcomes"]:
                        abr = NOMES_PARA_ABR.get(o["name"])
                        if abr == abr_casa:
                            odd_casa = o["price"]
                        elif abr == abr_vis:
                            odd_vis = o["price"]
                        else:
                            odd_empate = o["price"]

        if not odd_casa or not odd_vis:
            continue

        # Probabilidades implícitas normalizadas
        soma = (1/odd_casa) + (1/odd_vis) + (1/odd_empate if odd_empate else 0)
        prob_casa = round((1/odd_casa) / soma, 3) if soma else None
        prob_vis  = round((1/odd_vis)  / soma, 3) if soma else None

        def classificar(odd):
            if odd < 1.5:  return "favorito_forte"
            if odd < 2.0:  return "favorito"
            if odd < 2.5:  return "equilibrado"
            return "zebra"

        rows.append({
            "abr_casa":   abr_casa,
            "abr_vis":    abr_vis,
            "odd_casa":   odd_casa,
            "odd_vis":    odd_vis,
            "odd_empate": odd_empate,
            "prob_casa":  prob_casa,
            "prob_vis":   prob_vis,
            "forca_casa": classificar(odd_casa),
            "forca_vis":  classificar(odd_vis),
            "commence_time": jogo.get("commence_time"),
        })

    return pd.DataFrame(rows)


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


def enriquecer_partidas(df_partidas: pd.DataFrame, df_odds: pd.DataFrame, mapa_clubes: dict) -> pd.DataFrame:
    """Cruza partidas com odds usando mapa clube_id → abreviação."""
    df = df_partidas.copy()

    if df_odds.empty:
        return df

    col_casa = next((c for c in df.columns if "casa_id" in c), None)
    col_vis  = next((c for c in df.columns if "visitante_id" in c), None)

    if not col_casa or not col_vis:
        return df

    # Mapa abr → odds
    mapa_odds = {}
    for _, o in df_odds.iterrows():
        mapa_odds[o["abr_casa"]] = {
            "odd_casa":    o["odd_casa"],
            "odd_vis":     o["odd_vis"],
            "odd_empate":  o["odd_empate"],
            "prob_casa":   o["prob_casa"],
            "prob_vis":    o["prob_vis"],
            "forca_casa":  o["forca_casa"],
            "forca_vis":   o["forca_vis"],
        }

    def get_odd(clube_id, campo):
        try:
            abr = mapa_clubes.get(int(clube_id))
            return mapa_odds.get(abr, {}).get(campo)
        except Exception:
            return None

    df["odd_casa"]   = df[col_casa].apply(lambda x: get_odd(x, "odd_casa"))
    df["odd_vis"]    = df[col_vis].apply(lambda x: get_odd(x, "odd_vis"))
    df["odd_empate"] = df[col_casa].apply(lambda x: get_odd(x, "odd_empate"))
    df["prob_casa"]  = df[col_casa].apply(lambda x: get_odd(x, "prob_casa"))
    df["prob_vis"]   = df[col_vis].apply(lambda x: get_odd(x, "prob_vis"))
    df["forca_casa"] = df[col_casa].apply(lambda x: get_odd(x, "forca_casa"))
    df["forca_vis"]  = df[col_vis].apply(lambda x: get_odd(x, "forca_vis"))

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

# ── Mercado status ───────────────────────────────────────────
print("Extraindo status do mercado...")
try:
    raw_status = get_json("status")
    with open(DATA_DIR / "mercado_status.json", "w", encoding="utf-8") as f:
        json.dump(raw_status, f, ensure_ascii=False, indent=2)
    pd.DataFrame([raw_status]).to_csv(DATA_DIR / "mercado_status.csv", index=False, encoding="utf-8-sig")
    print(f"  OK")
    log.append({"endpoint": "status", "registros": 1, "status": "OK", "erro": ""})
except Exception as e:
    print(f"  ERRO: {e}")
    log.append({"endpoint": "status", "registros": 0, "status": "ERRO", "erro": str(e)})

# ── Odds ─────────────────────────────────────────────────────
print("Extraindo odds...")
df_odds = pd.DataFrame()
try:
    df_odds = get_odds()
    if not df_odds.empty:
        df_odds.to_csv(DATA_DIR / "odds.csv", index=False, encoding="utf-8-sig")
        print(f"  OK — {len(df_odds)} jogos")
        log.append({"endpoint": "odds", "registros": len(df_odds), "status": "OK", "erro": ""})
    else:
        print("  SKIP — sem dados de odds")
except Exception as e:
    print(f"  ERRO: {e}")
    log.append({"endpoint": "odds", "registros": 0, "status": "ERRO", "erro": str(e)})

# ── Partidas enriquecidas com odds ───────────────────────────
print("Enriquecendo partidas com odds...")
try:
    df_mercado  = dados_brutos.get("mercado", pd.DataFrame())
    df_partidas = dados_brutos.get("partidas", pd.DataFrame())

    # Mapa clube_id → abreviação a partir dos próprios atletas
    mapa_clubes = {}
    if not df_mercado.empty:
        for _, row in df_mercado.iterrows():
            if pd.notna(row.get("clube_id")) and pd.notna(row.get("clube")):
                mapa_clubes[int(row["clube_id"])] = row["clube"]

    if not df_partidas.empty and not df_odds.empty:
        df_partidas_enr = enriquecer_partidas(df_partidas, df_odds, mapa_clubes)
        df_partidas_enr.to_csv(DATA_DIR / "partidas.csv", index=False, encoding="utf-8-sig")
        print(f"  OK — odds cruzadas em {len(df_partidas_enr)} partidas")
    else:
        print("  SKIP — partidas ou odds vazias")
except Exception as e:
    print(f"  ERRO: {e}")
    log.append({"endpoint": "partidas_odds", "registros": 0, "status": "ERRO", "erro": str(e)})

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
ts = datetime.now(BRT).strftime("%d/%m/%Y %H:%M")
log_df = pd.DataFrame(log)
log_df.insert(0, "timestamp", ts)

log_path = DATA_DIR / "log.csv"
if log_path.exists():
    log_existente = pd.read_csv(log_path)
    log_df = pd.concat([log_existente, log_df], ignore_index=True)

log_df.to_csv(log_path, index=False, encoding="utf-8-sig")
print(f"\nExtração concluída — {ts}")