"""
cartola_extractor.py
--------------------
Extrai dados da API do Cartola FC, enriquece com colunas calculadas
e salva em data/. Roda via GitHub Actions diariamente.

Colunas extras geradas em atletas_enriquecido.csv:
  - mandante        : True se o clube joga em casa nessa rodada
  - adversario      : abreviação do adversário na rodada
  - tendencia       : 'alta' | 'baixa' | 'estavel' com base na variação
  - custo_beneficio : media / preco
  - cb_rank         : posição no ranking de custo-benefício dentro da posição
  - armadilha       : True se preço acima da mediana mas média abaixo da mediana da posição
  - status_label    : texto legível do status (Provável, Dúvida, etc.)

Também coleta dados do Brasileirão via football-data.org e salva:
  - brasileirao_data.json  : tabela, artilheiros, histórico e aproveitamento casa/fora
  - brasileirao_tabela.csv : tabela completa planificada para análise externa
"""

import json
import os
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone, timedelta
from collections import defaultdict

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

FOOTBALL_DATA_KEY = os.environ.get("FOOTBALL_DATA_KEY", "")
FOOTBALL_DATA_URL = "https://api.football-data.org/v4"
BRASILEIRAO_ID    = "BSA"

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
# REQUISIÇÃO CARTOLA
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
        "apiKey":     ODDS_API_KEY,
        "regions":    "eu",
        "markets":    "h2h",
        "oddsFormat": "decimal",
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

        soma      = (1/odd_casa) + (1/odd_vis) + (1/odd_empate if odd_empate else 0)
        prob_casa = round((1/odd_casa) / soma, 3) if soma else None
        prob_vis  = round((1/odd_vis)  / soma, 3) if soma else None

        def classificar(odd):
            if odd < 1.5: return "favorito_forte"
            if odd < 2.0: return "favorito"
            if odd < 2.5: return "equilibrado"
            return "zebra"

        rows.append({
            "abr_casa":      abr_casa,
            "abr_vis":       abr_vis,
            "odd_casa":      odd_casa,
            "odd_vis":       odd_vis,
            "odd_empate":    odd_empate,
            "prob_casa":     prob_casa,
            "prob_vis":      prob_vis,
            "forca_casa":    classificar(odd_casa),
            "forca_vis":     classificar(odd_vis),
            "commence_time": jogo.get("commence_time"),
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────
# FOOTBALL-DATA.ORG — BRASILEIRÃO
# ─────────────────────────────────────────────────────────────

def _fd_get(path: str, params: dict = None) -> dict:
    url  = f"{FOOTBALL_DATA_URL}{path}"
    resp = requests.get(
        url,
        headers={"X-Auth-Token": FOOTBALL_DATA_KEY},
        params=params or {},
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()


def _aprov(w: int, d: int, l: int) -> float:
    total = w + d + l
    return round((w * 3 + d) / (total * 3) * 100, 1) if total else 0.0


def build_team_history(matches: list) -> dict:
    """
    Para cada time, monta o histórico cronológico de resultados:
    W = vitória, L = derrota, D = empate.
    """
    history = defaultdict(list)
    finished = sorted(
        [m for m in matches if m.get("status") == "FINISHED"],
        key=lambda m: m.get("utcDate", ""),
    )
    for match in finished:
        home = match["homeTeam"]["name"]
        away = match["awayTeam"]["name"]
        gh   = match["score"]["fullTime"].get("home")
        ga   = match["score"]["fullTime"].get("away")
        if gh is None or ga is None:
            continue
        if gh > ga:
            history[home].append("W")
            history[away].append("L")
        elif gh < ga:
            history[home].append("L")
            history[away].append("W")
        else:
            history[home].append("D")
            history[away].append("D")
    return dict(history)


def build_team_stats(matches: list) -> dict:
    """
    Aproveitamento separado por mando de campo: gols, vitórias, empates, derrotas.
    """
    stats = defaultdict(lambda: {
        "home": {"gf": 0, "ga": 0, "w": 0, "d": 0, "l": 0},
        "away": {"gf": 0, "ga": 0, "w": 0, "d": 0, "l": 0},
    })
    for match in matches:
        if match.get("status") != "FINISHED":
            continue
        home = match["homeTeam"]["name"]
        away = match["awayTeam"]["name"]
        gh   = match["score"]["fullTime"].get("home")
        ga   = match["score"]["fullTime"].get("away")
        if gh is None or ga is None:
            continue

        stats[home]["home"]["gf"] += gh
        stats[home]["home"]["ga"] += ga
        stats[away]["away"]["gf"] += ga
        stats[away]["away"]["ga"] += gh

        if gh > ga:
            stats[home]["home"]["w"] += 1
            stats[away]["away"]["l"] += 1
        elif gh < ga:
            stats[home]["home"]["l"] += 1
            stats[away]["away"]["w"] += 1
        else:
            stats[home]["home"]["d"] += 1
            stats[away]["away"]["d"] += 1

    return {k: dict(v) for k, v in stats.items()}


def build_tabela_csv(tabela: list, history: dict, team_stats: dict) -> pd.DataFrame:
    """
    Monta DataFrame plano da tabela do Brasileirão para análise e CSV.

    Colunas:
      posicao, time, pts, j, v, e, d, gp, gc, sg, aprov_pct,
      casa_v, casa_e, casa_d, casa_gp, casa_gc, casa_aprov_pct,
      fora_v, fora_e, fora_d, fora_gp, fora_gc, fora_aprov_pct,
      forma (últimos 5 resultados separados por vírgula, ex: W,L,W,D,W)
    """
    rows = []
    for entry in tabela:
        nome = entry.get("team", {}).get("name", "")
        w    = entry.get("won",  0)
        d    = entry.get("draw", 0)
        l    = entry.get("lost", 0)

        ts = team_stats.get(nome, {})
        h  = ts.get("home", {})
        a  = ts.get("away", {})

        hist  = history.get(nome, [])
        forma = ",".join(hist[-5:]) if hist else ""

        rows.append({
            "posicao":        entry.get("position"),
            "time":           nome,
            "pts":            entry.get("points"),
            "j":              entry.get("playedGames"),
            "v":              w,
            "e":              d,
            "d":              l,
            "gp":             entry.get("goalsFor"),
            "gc":             entry.get("goalsAgainst"),
            "sg":             entry.get("goalDifference"),
            "aprov_pct":      _aprov(w, d, l),
            "casa_v":         h.get("w", 0),
            "casa_e":         h.get("d", 0),
            "casa_d":         h.get("l", 0),
            "casa_gp":        h.get("gf", 0),
            "casa_gc":        h.get("ga", 0),
            "casa_aprov_pct": _aprov(h.get("w", 0), h.get("d", 0), h.get("l", 0)),
            "fora_v":         a.get("w", 0),
            "fora_e":         a.get("d", 0),
            "fora_d":         a.get("l", 0),
            "fora_gp":        a.get("gf", 0),
            "fora_gc":        a.get("ga", 0),
            "fora_aprov_pct": _aprov(a.get("w", 0), a.get("d", 0), a.get("l", 0)),
            "forma":          forma,
        })
    return pd.DataFrame(rows)


def get_brasileirao_data() -> dict:
    """
    Coleta tabela, partidas e artilheiros do Brasileirão via football-data.org.
    Retorna dicionário pronto para serializar como brasileirao_data.json.
    Também salva brasileirao_tabela.csv com dados planificados para análise.
    """
    if not FOOTBALL_DATA_KEY:
        print("  SKIP — FOOTBALL_DATA_KEY não definida")
        return {}

    standings_raw = _fd_get(f"/competitions/{BRASILEIRAO_ID}/standings")
    tables        = {t["type"]: t["table"] for t in standings_raw.get("standings", [])}
    tabela_total  = tables.get("TOTAL", [])

    matches_raw = _fd_get(f"/competitions/{BRASILEIRAO_ID}/matches")
    matches     = matches_raw.get("matches", [])

    scorers_raw = _fd_get(f"/competitions/{BRASILEIRAO_ID}/scorers", {"limit": 15})
    scorers     = scorers_raw.get("scorers", [])

    history    = build_team_history(matches)
    team_stats = build_team_stats(matches)

    # CSV plano para análise externa
    df_tabela = build_tabela_csv(tabela_total, history, team_stats)
    df_tabela.to_csv(DATA_DIR / "brasileirao_tabela.csv", index=False, encoding="utf-8-sig")
    print(f"  brasileirao_tabela.csv salvo — {len(df_tabela)} times")

    return {
        "tabela":          tabela_total,
        "artilheiros":     scorers,
        "historico_times": history,
        "team_stats":      team_stats,
    }


# ─────────────────────────────────────────────────────────────
# NORMALIZADORES
# ─────────────────────────────────────────────────────────────

def normalizar_mercado(raw: dict) -> pd.DataFrame:
    atletas  = raw.get("atletas", [])
    clubes   = {int(k): v.get("abreviacao", k) for k, v in raw.get("clubes",   {}).items()}
    posicoes = {int(k): v.get("nome", k)        for k, v in raw.get("posicoes", {}).items()}

    rows = []
    for a in atletas:
        scouts = a.get("scout") or {}
        rows.append({
            "atleta_id":     a.get("atleta_id"),
            "nome":          a.get("apelido", a.get("nome")),
            "clube_id":      a.get("clube_id"),
            "clube":         clubes.get(int(a.get("clube_id", 0)), a.get("clube_id")),
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
        col_casa = next((c for c in df_partidas.columns if "casa_id"      in c), None)
        col_vis  = next((c for c in df_partidas.columns if "visitante_id" in c), None)
        mapa_abr = {}
        if not df_mercado.empty:
            for _, row in df_mercado.iterrows():
                if pd.notna(row.get("clube_id")) and pd.notna(row.get("clube")):
                    mapa_abr[int(row["clube_id"])] = row["clube"]

        for _, p in df_partidas.iterrows():
            try:
                id_casa  = int(p[col_casa]) if col_casa else None
                id_vis   = int(p[col_vis])  if col_vis  else None
                abr_casa = mapa_abr.get(id_casa, str(id_casa))
                abr_vis  = mapa_abr.get(id_vis,  str(id_vis))
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

    df["mandante"]   = df["clube_id"].apply(lambda x: get_confronto(x, "mandante",  None))
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
    print("  OK")
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
        df_enriquecido = df_enriquecido[
            (df_enriquecido["status_id"].astype(str).isin(["7", "2"])) &
            (df_enriquecido["preco"] > 0)
        ]
        df_enriquecido.to_csv(DATA_DIR / "atletas_enriquecido.csv", index=False, encoding="utf-8-sig")
        # ── CSV enxuto para escalação ────────────────────────────────
        print("Gerando CSV enxuto para escalação...")
        try:
            df_partidas_validas = dados_brutos.get("partidas", pd.DataFrame())

            # times com jogo válido na rodada
            times_validos = set()
            if not df_partidas_validas.empty and "valida" in df_partidas_validas.columns:
                col_casa = next((c for c in df_partidas_validas.columns if "casa_id" in c), None)
                col_vis  = next((c for c in df_partidas_validas.columns if "visitante_id" in c), None)
                validas  = df_partidas_validas[df_partidas_validas["valida"].astype(str).str.lower() == "true"]
                if col_casa: times_validos.update(validas[col_casa].astype(str).tolist())
                if col_vis:  times_validos.update(validas[col_vis].astype(str).tolist())

            cols_enxuto = [
                "nome", "clube", "clube_id", "posicao", "status_id", "status_label",
                "preco", "media", "variacao", "jogos",
                "custo_beneficio", "cb_rank",
                "mandante", "adversario", "armadilha",
            ]
            cols_enxuto = [c for c in cols_enxuto if c in df_enriquecido.columns]

            df_enxuto = df_enriquecido[
                df_enriquecido["status_id"].astype(str).isin(["7", "2"]) &
                (df_enriquecido["preco"] > 0) &
                (df_enriquecido["clube_id"].astype(str).isin(times_validos))
            ][cols_enxuto].sort_values("media", ascending=False)

            df_enxuto.to_csv(DATA_DIR / "atletas_escalacao.csv", index=False, encoding="utf-8-sig")
            print(f"  OK — {len(df_enxuto)} atletas no enxuto (de {len(df_enriquecido)} total)")
            log.append({"endpoint": "escalacao", "registros": len(df_enxuto), "status": "OK", "erro": ""})
        except Exception as e:
            print(f"  ERRO no enxuto: {e}")
            log.append({"endpoint": "escalacao", "registros": 0, "status": "ERRO", "erro": str(e)})

        print(f"  OK — {len(df_enriquecido)} atletas enriquecidos")
        log.append({"endpoint": "enriquecido", "registros": len(df_enriquecido), "status": "OK", "erro": ""})
    else:
        print("  SKIP — mercado vazio")

except Exception as e:
    print(f"  ERRO no enriquecimento: {e}")
    log.append({"endpoint": "enriquecido", "registros": 0, "status": "ERRO", "erro": str(e)})

# ── Brasileirão ──────────────────────────────────────────────
print("Extraindo tabela do Brasileirão...")
try:
    bra_data = get_brasileirao_data()
    if bra_data:
        with open(DATA_DIR / "brasileirao_data.json", "w", encoding="utf-8") as f:
            json.dump(bra_data, f, ensure_ascii=False, indent=2)
        n_times = len(bra_data.get("tabela", []))
        print(f"  OK — {n_times} times · brasileirao_data.json + brasileirao_tabela.csv")
        log.append({"endpoint": "brasileirao", "registros": n_times, "status": "OK", "erro": ""})
    else:
        print("  SKIP — FOOTBALL_DATA_KEY não definida ou sem dados")
        log.append({"endpoint": "brasileirao", "registros": 0, "status": "SKIP", "erro": "chave ausente"})
except Exception as e:
    print(f"  ERRO: {e}")
    log.append({"endpoint": "brasileirao", "registros": 0, "status": "ERRO", "erro": str(e)})

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