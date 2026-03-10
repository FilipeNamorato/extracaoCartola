import requests

API_KEY = ""

url = "https://api.the-odds-api.com/v4/sports/soccer_brazil_campeonato/odds"
params = {
    "apiKey": API_KEY,
    "regions": "eu",
    "markets": "h2h",
    "oddsFormat": "decimal"
}

response = requests.get(url, params=params)
jogos = response.json()

for jogo in jogos:
    print(f"\n{jogo['home_team']} vs {jogo['away_team']}")
    print(f"Data: {jogo['commence_time']}")
    for bookmaker in jogo['bookmakers'][:1]:
        for outcome in bookmaker['markets'][0]['outcomes']:
            print(f"  {outcome['name']}: {outcome['price']}")