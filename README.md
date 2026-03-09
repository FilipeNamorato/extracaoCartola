# cartola-data

Coleta automática de dados da API do Cartola FC via GitHub Actions.

## Arquivos gerados em `data/`

| Arquivo | Conteúdo |
|---|---|
| `atletas_mercado.csv` | Todos os atletas do mercado com preço, média, variação e scouts |
| `atletas_mercado.json` | JSON bruto do endpoint `/atletas/mercado` |
| `atletas_pontuados.csv` | Atletas que pontuaram na rodada atual |
| `atletas_pontuados.json` | JSON bruto do endpoint `/atletas/pontuados` |
| `partidas.csv` | Partidas da rodada |
| `partidas.json` | JSON bruto do endpoint `/partidas` |
| `rodadas.csv` | Histórico de rodadas |
| `rodadas.json` | JSON bruto do endpoint `/rodadas` |
| `log.csv` | Histórico de execuções com status e timestamp |

## Agendamento

Coleta automática todo dia às **8h (horário de Brasília)** via GitHub Actions.

Para rodar manualmente: `Actions > Coleta Cartola FC > Run workflow`

## Uso local

```bash
pip install requests pandas
python cartola_extractor.py
```
