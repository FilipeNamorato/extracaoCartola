# cartola-data

Coleta automática de dados da API do Cartola FC via GitHub Actions, com dashboard visual hospedado no GitHub Pages.

## Dashboard

Acesse em: **https://filipenamorato.github.io/extracaoCartola/**

## Arquivos gerados em `docs/data/`

| Arquivo | Conteúdo |
| --- | --- |
| `atletas_mercado.csv` | Todos os atletas do mercado com preço, média, variação e scouts |
| `atletas_mercado.json` | JSON bruto do endpoint `/atletas/mercado` |
| `atletas_pontuados.csv` | Atletas que pontuaram na rodada atual |
| `atletas_pontuados.json` | JSON bruto do endpoint `/atletas/pontuados` |
| `atletas_enriquecido.csv` | Atletas com colunas extras: mandante, adversário, tendência, custo-benefício, rank, armadilha |
| `partidas.csv` | Partidas da rodada |
| `partidas.json` | JSON bruto do endpoint `/partidas` |
| `rodadas.csv` | Histórico de rodadas |
| `rodadas.json` | JSON bruto do endpoint `/rodadas` |
| `mercado_status.csv` | Status atual do mercado com rodada atual e horário de fechamento |
| `mercado_status.json` | JSON bruto do endpoint `/mercado/status` |
| `log.csv` | Histórico de execuções com status e timestamp |

## Colunas extras em `atletas_enriquecido.csv`

| Coluna | Descrição |
| --- | --- |
| `mandante` | True se o clube joga em casa nessa rodada |
| `adversario` | Abreviação do adversário na rodada |
| `tendencia` | alta / baixa / estavel com base na variação |
| `custo_beneficio` | média ÷ preço |
| `cb_rank` | Ranking de custo-benefício dentro da posição |
| `armadilha` | True se preço acima da mediana mas média abaixo |
| `status_label` | Texto legível do status (Provável, Dúvida, etc.) |

## Agendamento

Coleta automática 3 vezes ao dia via GitHub Actions:

| Horário BRT | Horário UTC |
| --- | --- |
| 08h00 | 11h00 |
| 13h00 | 16h00 |
| 18h00 | 21h00 |

Para rodar manualmente: `Actions > Coleta Cartola FC > Run workflow`

## Uso local

```bash
pip install requests pandas
python cartola_extractor.py
```

Os arquivos serão gerados em `docs/data/`.