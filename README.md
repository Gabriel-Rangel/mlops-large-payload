# mlops-large-payload

Projeto de referência para evoluir um **POC de notebook** (modelo no Unity Catalog + Model Serving) até um fluxo de **MLOps**, com payloads **maiores que 1 MiB**.

Caso do cliente: **15 variáveis escalares** + **1 campo JSON ≈ 16 MB**. A demo usa **~15 MiB** por request.

Workspace em que isto foi implantado: `https://fevm-gabriel-caballeria-demo.cloud.databricks.com/` (profile CLI `fevm-gabriel`). Catálogo de demo: `gabriel_caballeria_demo_catalog`. Schema: `mlops_large_payload`.

## O problema

| Necessidade | Limitação |
|-------------|-----------|
| Governar modelos, ciclo de vida e promoção | Notebook ad-hoc não escala (sem CI/CD, aliases, gates, permissões) |
| Observabilidade de inferência | Inference tables do AI Gateway **não** guardam payloads > ~1 MiB |
| Serving online com JSON ~16 MB | Custom Model Serving aceita até **~16 MB no body HTTP**. JSON + envelope das 15 vars fica no limite |
| Serving ler o Volume | O container do Model Serving **não monta** `/Volumes`. Leitura é via **UC Files API**, com `DATABRICKS_HOST` + token |

## O que este repositório entrega

Duas abordagens **implementadas e testadas** com JSON de 15 MiB:

| | A — Model Serving | B — Databricks App (API assíncrona) |
|---|---|---|
| Como o JSON chega | Cliente grava no Volume **antes**. O POST ao endpoint é pequeno (`payload_uri` + 15 vars) | Cliente manda o JSON **no POST**. O App grava no Volume e devolve **202** + `request_id` |
| Como o modelo usa o JSON | PyFunc lê o Volume (Files API), deriva features de `items`, pontua sklearn | App pontua o Champion **no processo** (não chama o endpoint) |
| Onde o JSON fica | Volume `request_payloads` | **O mesmo** Volume |
| Observabilidade | Cliente (notebook/job) insere a linha | O App insere a linha |
| `endpoint_name` na tabela | `large-payload-scorer-dev` | `large-payload-api` |

O modelo **usa o conteúdo** do JSON: lê `payload["items"]` e deriva `json_n_items`, `json_total_amount`, `json_avg_amount`, `json_max_amount`, `json_unique_categories`. O padding de 15 MiB (`meta.pad`) **não** entra no score.

Também há:

1. **Databricks Asset Bundle** com targets `dev` / `staging` / `prod`.
2. **MLflow + UC**: aliases `Challenger` → `Champion`, tag `validation_status=valid`.
3. **Jobs** de setup, treino, promoção, deploy, demo, batch, validação e curl.
4. Tabela **`inference_observability`** (metadados + ponteiro; sem JSON completo).

```
Git → CI → Train (dev) + register UC → Evaluate gates
  → Champion → Deploy Serving  e/ou  Databricks App
  → Volume + inference_observability
```

## Estrutura

```
databricks.yml                      Bundle + variáveis por target
resources/
  uc_assets.yml                     Schema + Volumes
  jobs.yml                          Jobs do ciclo de vida
  inference_api.app.yml             Databricks App large-payload-api
src/
  model/                            PyFunc + derive_features_from_json
  client/                           Grava Volume + record do Serving
  app/                              FastAPI assíncrona (score local)
  notebooks/
    01_setup_uc.py                  DDL Volume + observability + grants
    02_train_register.py            Treino + Challenger
    03_evaluate_promote.py          Gates + Champion
    04_deploy_endpoint.py           Model Serving + env Files API
    05_demo_invoke.py               Serving com JSON ~15 MiB
    06_async_batch.py               Fila batch @Champion
    07_validate_approaches.py       Pass/fail A + B (0.01 / 2 / 15 MiB)
    08_curl_15mb.py                 Um JSON de 15 MiB **no body** do curl (Serving e App)
docs/
  ARCHITECTURE.md
  CUSTOMER_RESPONSE.md
tests/test_features.py              Testes offline
conf/model_tags.yml
```

## Tabela `inference_observability`

| Coluna | Conteúdo |
|--------|----------|
| `request_id` | Id da chamada |
| `payload_uri` | `/Volumes/<cat>/<schema>/request_payloads/dt=YYYY-MM-DD/<id>.json` |
| `payload_hash` / `payload_bytes` | Integridade e tamanho. Hash errado no Serving → HTTP 400 |
| `scalars_json` | As 15 variáveis. **Não** o JSON grande |
| `accept_latency_ms` | Envio + geração do `request_id` (Volume+id no Serving; 202 no App) |
| `total_latency_ms` | Até o resultado. `latency_ms` é alias desta coluna |
| `prediction` / `probability` / `status` | Resultado |
| `endpoint_name` | Distingue Serving vs App |

O JSON completo **não** vai nesta tabela. Só no Volume.

## Passo a passo

Não escolha profile sozinho. Liste e passe o nome no `--profile`:

```bash
databricks auth profiles
# Este workspace de demo: --profile fevm-gabriel
```

### 1. Pré-requisitos

- Databricks CLI ≥ 1.0 (`databricks -v`)
- Profile autenticado no workspace
- Permissão para schema, volume, modelo UC, jobs, Model Serving e Apps
- SQL warehouse (variável `warehouse_id` no `databricks.yml`; demo: `bf9351ab7a73e322`)
- Catálogo UC. No target `dev` deste repo: `gabriel_caballeria_demo_catalog`. Em outro cliente, altere `databricks.yml` (`ml_dev` / `ml_staging` / `ml_prod` ou o catálogo real)

### 2. Secret scope (Serving precisa disto para ler o Volume)

O endpoint **não** monta `/Volumes`. O PyFunc usa `WorkspaceClient().files.download`, então o serving precisa de host + token.

```bash
# Scope (uma vez)
databricks secrets create-scope mlops-large-payload --profile <PROFILE>

# PAT com permissão de ler o Volume. Não commite e não imprima o valor.
databricks secrets put-secret mlops-large-payload databricks-token --profile <PROFILE>
```

O job `deploy_endpoint` injeta no serving:

- `DATABRICKS_HOST` = URL do workspace
- `DATABRICKS_TOKEN` = `{{secrets/mlops-large-payload/databricks-token}}` (referência literal, não o valor)

O identity do serving (e o dono do PAT) precisa de, **um GRANT por statement**:

- `USE CATALOG` no catálogo
- `USE SCHEMA` no schema
- `READ VOLUME` em `request_payloads`

### 3. Testes locais (sem workspace)

```bash
python -m unittest tests.test_features
```

### 4. Validar e deployar o bundle

```bash
databricks bundle validate -t dev --profile <PROFILE>
databricks bundle deploy -t dev --profile <PROFILE>
```

Isto cria/atualiza Jobs, Volumes (via `uc_assets.yml`) e a App `large-payload-api`. No `dev`, `presets.name_prefix` está vazio para o nome do modelo/job não quebrar.

Se o código da App mudou e o bundle marcar o recurso como unchanged:

```bash
databricks apps deploy large-payload-api \
  --profile <PROFILE> \
  --source-code-path "/Workspace/Users/<você>/.bundle/mlops-large-payload/dev/files/src/app"
```

### 5. Rodar o ciclo de vida (ordem)

```bash
databricks bundle run setup_governance -t dev --profile <PROFILE>
databricks bundle run train_register -t dev --profile <PROFILE>
databricks bundle run evaluate_promote -t dev --profile <PROFILE> --params promote_to_champion=true
databricks bundle run deploy_endpoint -t dev --profile <PROFILE>
```

O que cada job faz:

| Job | Notebook | Função |
|-----|----------|--------|
| `setup_governance` | `01_setup_uc` | Schema, Volumes, `inference_observability`, `async_inference_queue`, grants |
| `train_register` | `02_train_register` | Treina sklearn, registra PyFunc, alias **Challenger** |
| `evaluate_promote` | `03_evaluate_promote` | Métrica + signature + smoke no Volume; alias **Champion**; tag `validation_status=valid` |
| `deploy_endpoint` | `04_deploy_endpoint` | Cria/atualiza Serving `@Champion`; poll `READY` (falha se `UPDATE_FAILED`) |
| `demo_invoke` | `05_demo_invoke` | Um JSON ~15 MiB no Volume + POST Serving + linha de observabilidade |
| `async_batch_score` | `06_async_batch` | Fila Delta → score `@Champion` (schedule pausado) |
| `validate_approaches` | `07_validate_approaches` | Pass/fail Serving e App em 0,01 / 2 / 15 MiB, incluindo latências |
| `curl_15mb` | `08_curl_15mb` | **Um** JSON de 15 MiB **junto** com as 15 variáveis no POST (`curl`) para Serving e App |

### 6. Conferir as duas abordagens

```bash
databricks bundle run demo_invoke -t dev --profile <PROFILE>
databricks bundle run validate_approaches -t dev --profile <PROFILE>
databricks bundle run curl_15mb -t dev --profile <PROFILE>
```

App neste workspace:

`https://large-payload-api-7474644345582688.aws.databricksapps.com`

Serving neste workspace (dev): endpoint `large-payload-scorer-dev`.

### 7. OAuth M2M para chamar a App a partir de Jobs

Databricks Apps **rejeitam PAT** (redirecionam para login). Jobs usam um service principal:

1. SP `mlops-large-payload-validator` com **CAN_USE** na App
2. Secret OAuth no scope `mlops-large-payload`:
   - `validator-client-id`
   - `validator-client-secret`
3. O notebook gera token: `POST {host}/oidc/v1/token` (`grant_type=client_credentials`, `scope=all-apis`)

Os notebooks `07` e `08` já fazem isso. Sem esses secrets, o teste da App é SKIP (não FAIL).

Grants do **App service principal** (não o validator): `USE CATALOG`, `USE SCHEMA`, `READ VOLUME` / `WRITE VOLUME` em `request_payloads`, `MODIFY` em `inference_observability`, `EXECUTE` no modelo, `CAN_USE` no warehouse. O bundle declara isso em `resources/inference_api.app.yml`.

## Contratos HTTP

### A — Model Serving

Dois contratos:

- **Referência (`payload_uri`)** — notebooks `05` / `07`: JSON no Volume, body pequeno.
- **Inline (`payload` no POST)** — notebook `08` (curl): as 15 variáveis **e** o JSON de ~15 MiB no mesmo body.

```json
{
  "dataframe_records": [
    {
      "var_01": 0.01,
      "var_15": 0.15,
      "payload": { "items": [ { "amount": 1.23, "category": "PAD" } ], "meta": { "pad": "…" } },
      "payload_uri": "",
      "payload_hash": "",
      "request_id": "abc"
    }
  ]
}
```

```bash
curl -sS -X POST "$HOST/serving-endpoints/large-payload-scorer-dev/invocations" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  --data-binary @serving_body.json
```

Resposta esperada: HTTP **200** com `prediction` e `probability`. Hash divergente: **400**.

### B — App assíncrona (JSON no POST)

```json
{
  "var_01": 0.01,
  "var_15": 0.15,
  "payload": { "items": [ { "amount": 1.23, "category": "PAD" } ], "meta": { "pad": "…" } }
}
```

| Método | Rota | Papel |
|--------|------|--------|
| GET | `/health` | Modelo carregado? |
| POST | `/invocations` (alias `/score`) | **202** `{ request_id, status: queued, payload_uri, accept_latency_ms, status_url }` |
| POST | `/invocations/multipart` | Mesmo fluxo com arquivo |
| GET | `/invocations/{request_id}` | Poll até `done` / `failed` |

```bash
curl -sS -X POST "$APP_URL/invocations" \
  -H "Authorization: Bearer $OAUTH_TOKEN" \
  -H "Content-Type: application/json" \
  --data-binary @app_body.json

curl -sS "$APP_URL/invocations/$REQUEST_ID" \
  -H "Authorization: Bearer $OAUTH_TOKEN"
```

Limite de body da App: `MAX_BODY_BYTES` (default 20 MiB). 15 MiB passou nos testes.

Status da fila fica **em memória** no processo da App (some no restart). Volume + tabela permanecem.

## Como o modelo usa o payload

1. Lê o JSON (`payload_uri` no Serving; memória no App).
2. `derive_features_from_json` usa só `items[].amount` e `items[].category`.
3. Junta com `var_01`…`var_15` e chama `predict_proba`.

Sem `items`, as features do JSON viram zero e o score fica só nas 15 variáveis. O gerador de demo inclui um `items` mínimo e incha `meta.pad` até ~15 MiB.

## Targets do bundle

| Target | Catálogo (default deste repo) | Endpoint | Experimento |
|--------|-------------------------------|----------|-------------|
| `dev` | `gabriel_caballeria_demo_catalog` | `large-payload-scorer-dev` | `/Shared/mlops-large-payload-dev` |
| `staging` | `ml_staging` | `large-payload-scorer-staging` | `/Shared/mlops-large-payload/staging` |
| `prod` | `ml_prod` | `large-payload-scorer` | `/Shared/mlops-large-payload/prod` |

Ajuste catálogo, `warehouse_id`, grupos (`ml_engineers_group` / `ml_ops_group`) e host do Serving em `databricks.yml` / `jobs.yml` antes de levar a outro workspace.

## Arquitetura e texto ao cliente

- Decisões e trade-offs: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
- Texto para Slack/e-mail: [`docs/CUSTOMER_RESPONSE.md`](docs/CUSTOMER_RESPONSE.md)

## Próximos passos com o cliente

1. Inferência **síncrona** (Serving + `payload_uri`) vs **assíncrona** (App 202 + poll).
2. Mapear catálogos, workspaces e quem promove para prod.
3. Trocar `SCALAR_FEATURES` + `derive_features_from_json` pelo schema real do JSON.
4. Retenção/PII do Volume e da tabela de observabilidade.
5. CI (`databricks bundle deploy`) e rollback (Champion anterior + `deploy_endpoint`).
6. Substituir o job store in-memory da App por Delta/`async_inference_queue` se precisar sobreviver a restart.
