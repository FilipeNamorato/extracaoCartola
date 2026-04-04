"""
cartola_extractor.py
--------------------
Extrai dados da API do Cartola FC, enriquece com colunas calculadas
e salva em data/. Roda via GitHub Actions diariamente.

Colunas extras geradas em atletas_enriquecido.csv:
  - mandante              : True se o clube joga em casa nessa rodada
  - adversario            : abreviação do adversário na rodada
  - tendencia             : 'alta' | 'baixa' | 'estavel'
  - confiabilidade        : fator 0-1 baseado no número de jogos (penaliza amostras pequenas)
  - media_bayesiana       : média com encolhimento para a média da posição (Bayesian shrinkage)
  - residuo_z             : z-score do desvio entre média e o esperado pelo preço (via regressão)
  - armadilha_label       : armadilha_forte | armadilha_leve | neutro | valor_bom | valor_oculto
  - custo_beneficio       : media_bayesiana / preco × confiabilidade
  - cb_rank               : ranking dentro da posição por custo-benefício ajustado
  - status_label          : texto legível do status
  - time_pos              : posição do time na tabela do Brasileirão
  - adv_pos               : posição do adversário na tabela
  - vantagem_mando        : delta de aproveitamento no mando específico (pp)
  - oportunidade_confronto: percentil 0-1 de oportunidade dada a posição do atleta
  - time_momentum_of      : ratio gols marcados recentes / média da temporada (>1 = em alta)
  - time_momentum_def     : ratio gols sofridos recentes / média (<1 = defesa em alta)
  - adv_momentum_of       : momentum ofensivo do adversário
  - adv_momentum_def      : momentum defensivo do adversário
  - sequencia_time        : streak atual (+N = N jogos invicto, -N = N jogos sem ganhar)
  - forma_score_time      : pontuação de forma ponderada 0-1 (mais recente = mais peso)
  - score_confronto_z     : score composto do confronto, normalizado por posição
"""

import json
import os
import numpy as np
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

# Mapa de nomes da Odds API / football-data → abreviações do Cartola
NOMES_PARA_ABR = {
    "Flamengo":            "FLA",
    "Palmeiras":           "PAL",
    "Atletico Mineiro":    "CAM",
    "Atlético Mineiro":    "CAM",
    "Fluminense":          "FLU",
    "Corinthians":         "COR",
    "Sao Paulo":           "SAO",
    "São Paulo":           "SAO",
    "Internacional":       "INT",
    "Gremio":              "GRE",
    "Grêmio":              "GRE",
    "Botafogo":            "BOT",
    "Vasco da Gama":       "VAS",
    "Bahia":               "BAH",
    "Cruzeiro":            "CRU",
    "Atletico Paranaense": "CAP",
    "Atlético Paranaense": "CAP",
    "Santos":              "SAN",
    "Vitoria":             "VIT",
    "Vitória":             "VIT",
    "Bragantino-SP":       "RBB",
    "Red Bull Bragantino": "RBB",
    "Mirassol":            "MIR",
    "Chapecoense":         "CHA",
    "Coritiba":            "CFC",
    "Remo":                "REM",
    "Sport Recife":        "SPT",
    "Ceará":               "CEA",
    "Fortaleza":           "FOR",
    "Juventude":           "JUV",
    "América Mineiro":     "AME",
    "Goiás":               "GOI",
    "Cuiabá":              "CUI",
}

BRT = timezone(timedelta(hours=-3))

DATA_DIR = Path("docs/data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ── Parâmetros estatísticos ───────────────────────────────────
# K do Bayesian shrinkage: abaixo disso a média é considerada não confiável
JOGOS_CONFIANCA_PLENA = 8
# Janela de jogos para cálculo de momentum
JANELA_MOMENTUM = 5
# Pesos da forma ponderada (5 jogos, mais recente = maior peso)
PESOS_FORMA = [0.10, 0.15, 0.20, 0.25, 0.30]

# Posições para lógica de oportunidade de confronto
POSICOES_ATAQUE = {"Atacante", "Meia"}
POSICOES_DEFESA = {"Zagueiro", "Lateral", "Goleiro"}


# ─────────────────────────────────────────────────────────────
# HELPERS DE MAPEAMENTO
# ─────────────────────────────────────────────────────────────
# ── HELPERS DE MAPEAMENTO ─────────────────────────────────────

import unicodedata

def normalizar_nome(nome: str) -> str:
    """Remove acentos e padroniza para comparação tolerante."""
    return unicodedata.normalize("NFKD", nome).encode("ascii", "ignore").decode("ascii").lower().strip()

# Pré-computa o índice normalizado uma vez
_NOMES_NORM = {normalizar_nome(k): v for k, v in NOMES_PARA_ABR.items()}

def get_nomes_por_abr(abr: str) -> list:
    return [nome for nome, a in NOMES_PARA_ABR.items() if a == abr]

def get_tabela_row(abr: str, tabela_idx: pd.DataFrame):
    """
    Busca tolerante: tenta match exato primeiro, depois normalizado.
    Loga miss para facilitar diagnóstico.
    """
    for nome in get_nomes_por_abr(abr):
        if nome in tabela_idx.index:
            return tabela_idx.loc[nome]
    
    # Fallback: match normalizado contra o índice real do football-data
    abr_norm = normalizar_nome(abr)
    for idx_nome in tabela_idx.index:
        if normalizar_nome(idx_nome) in [normalizar_nome(n) for n in get_nomes_por_abr(abr)]:
            return tabela_idx.loc[idx_nome]
    
    # Log para detectar mapeamentos quebrados
    if abr and abr != "—":
        print(f"  [WARN] get_tabela_row: sem match para abr='{abr}' | nomes tentados={get_nomes_por_abr(abr)}")
    return None


def get_momentum_time(abr: str, momentum: dict) -> dict:
    """Busca tolerante com fallback normalizado."""
    for nome in get_nomes_por_abr(abr):
        if nome in momentum:
            return momentum[nome]
    # Fallback normalizado
    abr_norm_nomes = [normalizar_nome(n) for n in get_nomes_por_abr(abr)]
    for k in momentum:
        if normalizar_nome(k) in abr_norm_nomes:
            return momentum[k]
    return {}

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
    """Histórico de resultados (W/D/L) por time, em ordem cronológica."""
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
    """Aproveitamento separado por mando de campo."""
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


def build_team_momentum(matches: list, n: int = JANELA_MOMENTUM) -> dict:
    """
    Para cada time, calcula momentum ofensivo e defensivo com base nos
    últimos N jogos, usando os placares reais (não apenas W/L/D).

    Retorna por time:
      momentum_of          : ratio gols marcados (últimos N / temporada). >1 = time marcando mais.
      momentum_def         : ratio gols sofridos (últimos N / temporada). <1 = defesa mais sólida.
      forma_score          : 0-1 ponderado dos últimos 5 jogos (mais recente = 0.30).
      sequencia            : +N = N jogos invicto consecutivo, -N = N jogos sem ganhar.
      media_gf_temporada   : média de gols marcados na temporada.
      media_ga_temporada   : média de gols sofridos na temporada.
      media_gf_recente     : média de gols marcados nos últimos N jogos.
      media_ga_recente     : média de gols sofridos nos últimos N jogos.
    """
    finished = sorted(
        [m for m in matches if m.get("status") == "FINISHED"],
        key=lambda m: m.get("utcDate", ""),
    )

    # Série temporal de (gols_marcados, gols_sofridos) por time
    series = defaultdict(list)
    for match in finished:
        home = match["homeTeam"]["name"]
        away = match["awayTeam"]["name"]
        gh   = match["score"]["fullTime"].get("home")
        ga   = match["score"]["fullTime"].get("away")
        if gh is None or ga is None:
            continue
        series[home].append({"gf": gh, "ga": ga})
        series[away].append({"gf": ga, "ga": gh})

    momentum = {}
    for team, jogos in series.items():
        if not jogos:
            continue

        all_gf = [j["gf"] for j in jogos]
        all_ga = [j["ga"] for j in jogos]

        media_gf_temp = sum(all_gf) / len(all_gf)
        media_ga_temp = sum(all_ga) / len(all_ga)

        ultimos_n    = jogos[-n:]
        media_gf_rec = sum(j["gf"] for j in ultimos_n) / len(ultimos_n)
        media_ga_rec = sum(j["ga"] for j in ultimos_n) / len(ultimos_n)

        # Momentum: ratio recente / temporada inteira
        # >1 = crescendo, <1 = declinando
        momentum_of  = round(media_gf_rec / media_gf_temp, 3) if media_gf_temp > 0 else 1.0
        momentum_def = round(media_ga_rec / media_ga_temp, 3) if media_ga_temp > 0 else 1.0

        # Forma ponderada dos últimos 5 jogos
        ultimos5    = jogos[-5:]
        offset      = 5 - len(ultimos5)
        forma_score = 0.0
        for i, j in enumerate(ultimos5):
            peso = PESOS_FORMA[i + offset]
            if j["gf"] > j["ga"]:
                forma_score += peso * 1.0
            elif j["gf"] == j["ga"]:
                forma_score += peso * 0.5

        # Sequência: invicto consecutivo (positivo) ou sem ganhar (negativo)
        seq_invicto = 0
        for j in reversed(jogos):
            if j["gf"] >= j["ga"]:  # vitória ou empate
                seq_invicto += 1
            else:
                break

        if seq_invicto > 0:
            sequencia = seq_invicto
        else:
            seq_sem_ganhar = 0
            for j in reversed(jogos):
                if j["gf"] < j["ga"]:  # derrota
                    seq_sem_ganhar -= 1
                else:
                    break
            sequencia = seq_sem_ganhar

        momentum[team] = {
            "momentum_of":        momentum_of,
            "momentum_def":       momentum_def,
            "forma_score":        round(forma_score, 3),
            "sequencia":          sequencia,
            "media_gf_temporada": round(media_gf_temp, 2),
            "media_ga_temporada": round(media_ga_temp, 2),
            "media_gf_recente":   round(media_gf_rec, 2),
            "media_ga_recente":   round(media_ga_rec, 2),
        }

    return momentum


def build_tabela_csv(
    tabela: list,
    history: dict,
    team_stats: dict,
    momentum: dict = None,
) -> pd.DataFrame:
    """
    Monta DataFrame plano da tabela do Brasileirão para análise e CSV.
    Inclui métricas de momentum quando disponíveis.
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

        mom = (momentum or {}).get(nome, {})

        rows.append({
            # Tabela geral
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
            # Desempenho em casa
            "casa_v":         h.get("w", 0),
            "casa_e":         h.get("d", 0),
            "casa_d":         h.get("l", 0),
            "casa_gp":        h.get("gf", 0),
            "casa_gc":        h.get("ga", 0),
            "casa_aprov_pct": _aprov(h.get("w", 0), h.get("d", 0), h.get("l", 0)),
            # Desempenho fora
            "fora_v":         a.get("w", 0),
            "fora_e":         a.get("d", 0),
            "fora_d":         a.get("l", 0),
            "fora_gp":        a.get("gf", 0),
            "fora_gc":        a.get("ga", 0),
            "fora_aprov_pct": _aprov(a.get("w", 0), a.get("d", 0), a.get("l", 0)),
            # Forma
            "forma":                 forma,
            # Momentum (calculado dos placares reais)
            "momentum_of":           mom.get("momentum_of"),
            "momentum_def":          mom.get("momentum_def"),
            "forma_score":           mom.get("forma_score"),
            "sequencia":             mom.get("sequencia"),
            "media_gf_temporada":    mom.get("media_gf_temporada"),
            "media_ga_temporada":    mom.get("media_ga_temporada"),
            "media_gf_recente":      mom.get("media_gf_recente"),
            "media_ga_recente":      mom.get("media_ga_recente"),
        })
    return pd.DataFrame(rows)


def get_brasileirao_data() -> tuple:
    """
    Coleta tabela, partidas e artilheiros do Brasileirão via football-data.org.

    Retorna (dados_dict, df_tabela, momentum) onde:
      dados_dict : dicionário para brasileirao_data.json
      df_tabela  : DataFrame plano da tabela (já salvo em CSV)
      momentum   : dict de momentum por time (para enriquecimento dos atletas)
    """
    if not FOOTBALL_DATA_KEY:
        print("  SKIP — FOOTBALL_DATA_KEY não definida")
        return {}, pd.DataFrame(), {}

    standings_raw = _fd_get(f"/competitions/{BRASILEIRAO_ID}/standings")
    tables        = {t["type"]: t["table"] for t in standings_raw.get("standings", [])}
    tabela_total  = tables.get("TOTAL", [])

    matches_raw = _fd_get(f"/competitions/{BRASILEIRAO_ID}/matches")
    matches     = matches_raw.get("matches", [])

    scorers_raw = _fd_get(f"/competitions/{BRASILEIRAO_ID}/scorers", {"limit": 15})
    scorers     = scorers_raw.get("scorers", [])

    history    = build_team_history(matches)
    team_stats = build_team_stats(matches)
    momentum   = build_team_momentum(matches)

    df_tabela = build_tabela_csv(tabela_total, history, team_stats, momentum)
    df_tabela.to_csv(DATA_DIR / "brasileirao_tabela.csv", index=False, encoding="utf-8-sig")
    print(f"  brasileirao_tabela.csv salvo — {len(df_tabela)} times")

    dados = {
        "tabela":          tabela_total,
        "artilheiros":     scorers,
        "historico_times": history,
        "team_stats":      team_stats,
    }

    return dados, df_tabela, momentum


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
# ENRIQUECIMENTO CARTOLA
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

    # ── Tipos numéricos ──────────────────────────────────────
    df["preco"]    = pd.to_numeric(df["preco"],    errors="coerce").fillna(0)
    df["media"]    = pd.to_numeric(df["media"],    errors="coerce").fillna(0)
    df["jogos"]    = pd.to_numeric(df["jogos"],    errors="coerce").fillna(0)
    df["variacao"] = pd.to_numeric(df["variacao"], errors="coerce").fillna(0)

    # ── Confiabilidade ───────────────────────────────────────
    # Fator 0-1: atletas com menos de JOGOS_CONFIANCA_PLENA jogos têm peso menor.
    df["confiabilidade"] = (df["jogos"] / JOGOS_CONFIANCA_PLENA).clip(upper=1.0).round(3)

    # ── Média bayesiana (Bayesian shrinkage) ─────────────────
    # Puxa a média individual para a média da posição quando há poucos jogos.
    # media_bayes = (jogos × media + K × media_posicao) / (jogos + K)
    # Com muitos jogos, tende para a média bruta. Com poucos, para a média da posição.
    media_prior = df[df["jogos"] >= 3].groupby("posicao")["media"].mean()

    def calcular_media_bayesiana(row):
        j = row["jogos"]
        if j < 1:
            return 0.0
        prior = media_prior.get(row["posicao"], row["media"])
        return round(
            (j * row["media"] + JOGOS_CONFIANCA_PLENA * prior) / (j + JOGOS_CONFIANCA_PLENA), 3
        )

    df["media_bayesiana"] = df.apply(calcular_media_bayesiana, axis=1)

    # ── Resíduo z-score: armadilha por regressão linear ─────
    # Dentro de cada posição: regressão preco → media_bayesiana.
    # O resíduo mede o quanto o atleta entrega ACIMA ou ABAIXO do esperado pelo preço.
    # residuo_z > 0  : entrega mais do que o preço sugere (valor oculto)
    # residuo_z < 0  : entrega menos do que o preço sugere (armadilha)
    # Vantagem sobre o booleano: captura magnitude e não depende de mediana simples.
    residuos = []
    for pos, grp in df.groupby("posicao"):
        x = grp["preco"].values
        y = grp["media_bayesiana"].values
        if len(grp) < 3 or x.std() == 0:
            residuos.append(pd.Series(0.0, index=grp.index))
            continue
        coeffs = np.polyfit(x, y, 1)
        y_hat  = np.polyval(coeffs, x)
        resid  = y - y_hat
        std    = resid.std()
        z      = resid / std if std > 0 else np.zeros_like(resid)
        residuos.append(pd.Series(z.round(3), index=grp.index))

    df["residuo_z"] = pd.concat(residuos).reindex(df.index).fillna(0)

    def armadilha_label(z):
        if z < -1.5: return "armadilha_forte"
        if z < -0.5: return "armadilha_leve"
        if z >  1.5: return "valor_oculto"
        if z >  0.5: return "valor_bom"
        return "neutro"

    df["armadilha_label"] = df["residuo_z"].apply(armadilha_label)

    # ── Custo-benefício ajustado por confiabilidade ──────────
    df["custo_beneficio"] = df.apply(
        lambda r: round(r["media_bayesiana"] / r["preco"] * r["confiabilidade"], 3)
        if r["preco"] > 0 else 0,
        axis=1
    )
    df["cb_rank"] = (
        df[df["preco"] > 0]
        .groupby("posicao")["custo_beneficio"]
        .rank(ascending=False, method="min")
        .reindex(df.index)
        .fillna(0)
        .astype(int)
    )

    # ── Status legível ───────────────────────────────────────
    df["status_label"] = df["status_id"].apply(
        lambda x: STATUS_LABEL.get(int(x), "Desconhecido") if pd.notna(x) else "Desconhecido"
    )

    return df


def enriquecer_partidas(
    df_partidas: pd.DataFrame,
    df_odds: pd.DataFrame,
    mapa_clubes: dict,
) -> pd.DataFrame:
    """Cruza partidas com odds usando mapa clube_id → abreviação."""
    df = df_partidas.copy()

    if df_odds.empty:
        return df

    col_casa = next((c for c in df.columns if "casa_id"      in c), None)
    col_vis  = next((c for c in df.columns if "visitante_id" in c), None)

    if not col_casa or not col_vis:
        return df

    mapa_odds = {}
    for _, o in df_odds.iterrows():
        mapa_odds[o["abr_casa"]] = {
            "odd_casa":   o["odd_casa"],
            "odd_vis":    o["odd_vis"],
            "odd_empate": o["odd_empate"],
            "prob_casa":  o["prob_casa"],
            "prob_vis":   o["prob_vis"],
            "forca_casa": o["forca_casa"],
            "forca_vis":  o["forca_vis"],
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
# ENRIQUECIMENTO COM DADOS DO BRASILEIRÃO
# ─────────────────────────────────────────────────────────────

def enriquecer_com_confronto(
    df: pd.DataFrame,
    df_tabela: pd.DataFrame,
    momentum: dict,
) -> pd.DataFrame:
    """
    Cruza cada atleta com os dados do Brasileirão:

    - time_pos / adv_pos             : posição na tabela
    - vantagem_mando                 : delta de aproveitamento no mando específico do confronto
    - oportunidade_confronto         : percentil 0-1 de oportunidade dado a posição do atleta
        Atacantes/Meias : fragilidade defensiva do adversário no mando dele
        Defensores/GOL  : fraqueza ofensiva do adversário no mando dele
    - time_momentum_of/def           : ratio gols recentes / temporada para o time do atleta
    - adv_momentum_of/def            : idem para o adversário
    - forma_score_time               : forma ponderada do time (0-1)
    - sequencia_time                 : streak invicto (positivo) ou sem ganhar (negativo)
    - score_confronto_z              : score composto normalizado por posição
    """
    if df_tabela.empty:
        return df

    df = df.copy()

    # Pré-calcular gols por jogo (casa e fora) na tabela
    t = df_tabela.copy()
    t["j_casa"] = t["casa_v"] + t["casa_e"] + t["casa_d"]
    t["j_fora"] = t["fora_v"] + t["fora_e"] + t["fora_d"]

    t["gc_pg_casa"] = t.apply(
        lambda r: round(r["casa_gc"] / r["j_casa"], 3) if r["j_casa"] > 0 else 0, axis=1
    )
    t["gc_pg_fora"] = t.apply(
        lambda r: round(r["fora_gc"] / r["j_fora"], 3) if r["j_fora"] > 0 else 0, axis=1
    )
    t["gp_pg_casa"] = t.apply(
        lambda r: round(r["casa_gp"] / r["j_casa"], 3) if r["j_casa"] > 0 else 0, axis=1
    )
    t["gp_pg_fora"] = t.apply(
        lambda r: round(r["fora_gp"] / r["j_fora"], 3) if r["j_fora"] > 0 else 0, axis=1
    )

    # Percentis de fragilidade defensiva e força ofensiva
    # percentil_def_casa alto = adversário sofre muitos gols quando joga em casa (fraco defensivamente em casa)
    t["percentil_def_casa"] = t["gc_pg_casa"].rank(pct=True).round(3)
    t["percentil_def_fora"] = t["gc_pg_fora"].rank(pct=True).round(3)
    # percentil_of_casa alto = adversário marca muitos gols em casa (forte ofensivamente em casa)
    t["percentil_of_casa"]  = t["gp_pg_casa"].rank(pct=True).round(3)
    t["percentil_of_fora"]  = t["gp_pg_fora"].rank(pct=True).round(3)

    tabela_idx = t.set_index("time")

    results = []
    for _, atleta in df.iterrows():
        clube_abr = str(atleta.get("clube", ""))
        adv_abr   = str(atleta.get("adversario", ""))
        mandante  = atleta.get("mandante")
        posicao   = str(atleta.get("posicao", ""))

        row_time = get_tabela_row(clube_abr, tabela_idx)
        row_adv  = get_tabela_row(adv_abr,  tabela_idx)
        mom_time = get_momentum_time(clube_abr, momentum)
        mom_adv  = get_momentum_time(adv_abr,  momentum)

        rec = {
            "time_pos":               int(row_time["posicao"])  if row_time is not None else None,
            "adv_pos":                int(row_adv["posicao"])   if row_adv  is not None else None,
            "time_momentum_of":       mom_time.get("momentum_of"),
            "time_momentum_def":      mom_time.get("momentum_def"),
            "adv_momentum_of":        mom_adv.get("momentum_of"),
            "adv_momentum_def":       mom_adv.get("momentum_def"),
            "sequencia_time":         mom_time.get("sequencia"),
            "forma_score_time":       mom_time.get("forma_score"),
            "vantagem_mando":         None,
            "oportunidade_confronto": None,
        }

        # ── Vantagem de mando ──────────────────────────────
        # Delta entre o aproveitamento do time no mando em que joga
        # e o aproveitamento do adversário no mando contrário.
        # Positivo = o time tem vantagem estrutural nesse confronto específico.
        if row_time is not None and row_adv is not None and mandante is not None:
            if mandante:
                rec["vantagem_mando"] = round(
                    float(row_time["casa_aprov_pct"]) - float(row_adv["fora_aprov_pct"]), 1
                )
            else:
                rec["vantagem_mando"] = round(
                    float(row_time["fora_aprov_pct"]) - float(row_adv["casa_aprov_pct"]), 1
                )

        # ── Oportunidade de confronto por posição ──────────
        # Atacantes/Meias: quão fraca é a defesa do adversário no mando dele?
        #   Mandante=True  → adversário joga fora → usar percentil_def_fora do adv
        #   Mandante=False → adversário joga em casa → usar percentil_def_casa do adv
        # Defensores/GOL: quão fraco é o ataque do adversário no mando dele?
        #   (inverte: menor força ofensiva = melhor oportunidade para a defesa)
        if row_adv is not None and mandante is not None:
            if posicao in POSICOES_ATAQUE:
                rec["oportunidade_confronto"] = round(float(
                    row_adv["percentil_def_fora"] if mandante else row_adv["percentil_def_casa"]
                ), 3)
            elif posicao in POSICOES_DEFESA:
                rec["oportunidade_confronto"] = round(1.0 - float(
                    row_adv["percentil_of_fora"] if mandante else row_adv["percentil_of_casa"]
                ), 3)

        results.append(rec)

    for col in results[0].keys():
        df[col] = [r[col] for r in results]

    # ── Score composto do confronto ──────────────────────────
    # Componentes normalizados para 0-1 antes de combinar com pesos.
    # oportunidade_confronto : já é percentil 0-1
    # vantagem_mando         : clip em [-50, +50] e normaliza
    # time_momentum_of       : ratio clampado em [0.3, 2.0] e normalizado
    # forma_score_time       : já é 0-1
    # adv_em_queda           : inverso do momentum ofensivo do adversário

    # ── Score composto dinâmico por posição ──────────────────
    oc       = df["oportunidade_confronto"].fillna(0.5)
    vm       = ((df["vantagem_mando"].fillna(0).clip(-50, 50) + 50) / 100)
    tof_norm = ((df["time_momentum_of"].fillna(1.0).clip(0.3, 2.0) - 0.3) / 1.7)
    fs       = df["forma_score_time"].fillna(0.5)
    adv_norm = ((df["adv_momentum_of"].fillna(1.0).clip(0.3, 2.0) - 0.3) / 1.7)

    def calcular_score(row):
        # Defesa: Foco em não tomar gol (Vantagem de Mando e Queda do Adversário)
        if row["posicao"] in ["Zagueiro", "Lateral", "Goleiro"]:
            return (0.40 * row["oc"] + 0.30 * row["vm"] + 0.10 * row["tof_norm"] + 0.10 * row["fs"] + 0.10 * (1 - row["adv_norm"]))
        # Ataque: Foco em Momentum Ofensivo e Oportunidade
        else:
            return (0.35 * row["oc"] + 0.15 * row["vm"] + 0.30 * row["tof_norm"] + 0.15 * row["fs"] + 0.05 * (1 - row["adv_norm"]))

    df_temp = pd.DataFrame({'posicao': df['posicao'], 'oc': oc, 'vm': vm, 'tof_norm': tof_norm, 'fs': fs, 'adv_norm': adv_norm})
    df["score_confronto"] = df_temp.apply(calcular_score, axis=1).round(4)

    # 1. Isolar apenas quem vai jogar (não usar os nulos na estatística)
    mask_validos = (df["adversario"] != "—") & (df["adversario"].notna())

    # 2. Função matemática de Z-Score Seguro
    def z_score_seguro(x):
        std = x.std()
        if pd.isna(std) or std < 1e-6:
            # Sem variância real dentro do grupo: retorna 0 (mediano para todos)
            return pd.Series(0.0, index=x.index)
        # Clipa em 3 desvios para evitar outliers extremos
        return ((x - x.mean()) / std).clip(-3.0, 3.0)

    # 3. Calcula o Z-Score apenas na máscara válida e já clipa os excessos (-3 a +3)
    df["score_confronto_z"] = np.nan
    df.loc[mask_validos, "score_confronto_z"] = (
        df[mask_validos].groupby("posicao")["score_confronto"]
        .transform(z_score_seguro)
        .round(3)
        .clip(lower=-3.0, upper=3.0)
    )

    # 4. Transformação UX: Z-Score para "Nota Cartola" (0 a 100)
    # Z=0 (Mediano) vira Nota 50. Z=+3 (Excelente) vira Nota 95. Z=-3 (Fogueira) vira Nota 5.
    df["score_confronto_100"] = (50 + (df["score_confronto_z"] * 15)).round(1).clip(0, 100)

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

# ── Brasileirão (antes do enriquecimento de atletas) ─────────
# Movido para antes do CSV enriquecido para que momentum e tabela
# estejam disponíveis no momento do cruzamento com os atletas.
print("Extraindo tabela do Brasileirão...")
df_tabela_bra = pd.DataFrame()
momentum_bra  = {}
try:
    bra_data, df_tabela_bra, momentum_bra = get_brasileirao_data()
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

# ── CSV enriquecido ──────────────────────────────────────────
print("Gerando CSV enriquecido...")
try:
    df_mercado  = dados_brutos.get("mercado", pd.DataFrame())
    df_partidas = dados_brutos.get("partidas", pd.DataFrame())
    # ── DIAGNÓSTICO DE MAPEAMENTO ────────────────────────────────
    if not df_tabela_bra.empty:
        times_cartola = set(dados_brutos["mercado"]["clube"].dropna().unique())
        times_fd = set(df_tabela_bra["time"].dropna().unique())
        
        sem_match = []
        for abr in times_cartola:
            nomes = get_nomes_por_abr(abr)
            if not any(n in times_fd for n in nomes):
                sem_match.append(abr)
        
        if sem_match:
            print(f"  [WARN] Times sem match no football-data: {sem_match}")
            print(f"  Times disponíveis no football-data: {sorted(times_fd)}")
    if not df_mercado.empty:
        df_enriquecido = enriquecer(df_mercado, df_partidas)
        df_enriquecido = enriquecer_com_confronto(df_enriquecido, df_tabela_bra, momentum_bra)

        df_enriquecido = df_enriquecido[
            (df_enriquecido["status_id"].astype(str).isin(["7", "2"])) &
            (df_enriquecido["preco"] > 0)
        ]
        df_enriquecido.to_csv(DATA_DIR / "atletas_enriquecido.csv", index=False, encoding="utf-8-sig")

        # ── CSV enxuto para escalação ────────────────────────────
        print("Gerando CSV enxuto para escalação...")
        try:
            df_partidas_validas = dados_brutos.get("partidas", pd.DataFrame())

            times_validos = set()
            if not df_partidas_validas.empty and "valida" in df_partidas_validas.columns:
                col_casa = next((c for c in df_partidas_validas.columns if "casa_id"      in c), None)
                col_vis  = next((c for c in df_partidas_validas.columns if "visitante_id" in c), None)
                validas  = df_partidas_validas[df_partidas_validas["valida"].astype(str).str.lower() == "true"]
                if col_casa: times_validos.update(validas[col_casa].astype(str).tolist())
                if col_vis:  times_validos.update(validas[col_vis].astype(str).tolist())

            cols_enxuto = [
                # Identificação
                "nome", "clube", "clube_id", "posicao", "status_id", "status_label",
                # Preço e médias
                "preco", "media", "media_bayesiana", "variacao", "jogos",
                # Qualidade do dado
                "confiabilidade",
                # Custo-benefício
                "custo_beneficio", "cb_rank",
                # Contexto do confronto (Cartola)
                "mandante", "adversario", "tendencia",
                # Avaliação de valor
                "residuo_z", "armadilha_label",
                # Posição na tabela
                "time_pos", "adv_pos",
                # Vantagem estrutural
                "vantagem_mando", "oportunidade_confronto",
                # Momentum do time
                "time_momentum_of", "time_momentum_def",
                "forma_score_time", "sequencia_time",
                # Momentum do adversário
                "adv_momentum_of", "adv_momentum_def",
                # Score final
                "score_confronto_100",
            ]
            cols_enxuto = [c for c in cols_enxuto if c in df_enriquecido.columns]

            filtro_times = (
                df_enriquecido["clube_id"].astype(str).isin(times_validos)
                if times_validos else pd.Series(True, index=df_enriquecido.index)
            )

            df_enxuto = (
                df_enriquecido[
                    df_enriquecido["status_id"].astype(str).isin(["7", "2"]) &
                    (df_enriquecido["preco"] > 0) &
                    filtro_times
                ][cols_enxuto]
                .sort_values("score_confronto_z", ascending=False)
            )

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