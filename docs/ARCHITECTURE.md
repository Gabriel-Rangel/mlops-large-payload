# Arquitetura — MLOps com payload grande

## Contexto

O cliente já consegue:

- Treinar / registrar modelo no **Unity Catalog**
- Servir via **Model Serving endpoint**
- Fazer a ponte registro→serving em **notebook de POC**

Gaps para produção:

1. Governança e ciclo de vida (ambientes, aliases, gates, CI/CD, permissões)
2. Observabilidade quando o payload **excede ~1 MiB** (limite das inference tables)
3. Request com **JSON ≈ 16 MB** + 15 variáveis — no **limite de 16 MB** do Custom Model Serving

## Fluxo de MLOps (código + modelo)

```
Git (treino, eval, deploy, infra-as-code)
        │
        ▼
   CI: lint + unit tests + validate bundle
        │
        ▼
 ┌──────────────┐   register    ┌─────────────────────┐
 │ Job: train   │──────────────▶│ UC Model @Challenger│
 └──────────────┘               └──────────┬──────────┘
                                           │ gates
                                           ▼
                                ┌─────────────────────┐
                                │ UC Model @Champion  │
                                └──────────┬──────────┘
                                           │
                                           ▼
                                ┌─────────────────────┐
                                │ Model Serving EP    │
                                └──────────┬──────────┘
                                           │
                                           ▼
                                Observability + alerts
```

### Governança UC

| Prática | Como este projeto materializa |
|---------|--------------------------------|
| Catálogos por ambiente | Targets `dev` / `staging` / `prod`. Neste workspace de demo o `dev` usa `gabriel_caballeria_demo_catalog`; o YAML ainda cita `ml_staging` / `ml_prod` para os outros targets |
| Aliases em vez de “stage” legado | `Challenger` (candidato) e `Champion` (servido) |
| Sem user pessoal em prod | Jobs + service principals; grupos em `permissions` / grants |
| Tags | `owner`, `domain`, `criticality`, `payload_pattern`, `validation_status`, `roc_auc`, `code_ref` |
| Lineagem dados↔modelo | Registrar datasets com MLflow no Job de treino (estender `02_train_register`) |
| Promoção entre catálogos | `mlflow.copy_model_version` ou Job de promoção com SP dedicado |

### Infra como código

- **Databricks Asset Bundles** versionam Jobs, schema, Volumes e registered model.
- Terraform continua útil para account-level (metastore, storage credentials, SCIM) — complementar, não substituto do bundle de app ML.

## Payloads > 1 MiB / JSON ~16 MB

### Limites relevantes

| Camada | Limite típico | Implicação |
|--------|---------------|------------|
| Custom Model Serving request | ~16 MB | JSON de 16 MB + envelope das 15 vars → risco de falha |
| Inference tables (AI Gateway) | ~1 MiB / payload | Não use como store do JSON completo |
| Telemetry / custom logging rows | logs &lt; ~1 MB; records &lt; ~10 MB | Também não embutir o JSON |

### Padrão recomendado: referência + Volume

```
┌────────────┐  write JSON   ┌──────────────────────────┐
│ Consumidor │──────────────▶│ UC Volume request_payloads│
└─────┬──────┘               └────────────▲─────────────┘
      │ POST small body                   │ read by URI
      ▼                                   │
┌─────────────────────┐                   │
│ Model Serving       │───────────────────┘
│ (scalars+payload_uri)
└─────────┬───────────┘
          │ metadata row
          ▼
┌─────────────────────────────┐
│ Delta inference_observability│
│ request_id, hash, uri, ...  │
└─────────────────────────────┘
```

**Request de exemplo**

```json
{
  "var_01": 0.1,
  "var_15": 0.9,
  "payload_uri": "/Volumes/ml_prod/mlops_large_payload/request_payloads/dt=2026-09-25/uuid.json",
  "payload_hash": "sha256:…",
  "request_id": "uuid"
}
```

### Três opções de implementação

#### A — PyFunc lê o Volume (implementado em `src/model`)

- Custom Model Serving **não monta** `/Volumes` no filesystem do container. O PyFunc lê o JSON via **UC Files API** (`WorkspaceClient.files.download`) com as credenciais injetadas no serving.
- O identity do endpoint precisa de `USE CATALOG`, `USE SCHEMA` e `READ VOLUME`.
- Prós: contrato único; feature engineering versionada com o modelo.
- Contras: I/O na predição aumenta latência; 15 MiB por request é viável, mas não é o caminho de menor latência.

#### B — Pré-processamento / API fora do endpoint

**B1 — Databricks App assíncrona (implementada em `src/app`, App `large-payload-api`)**

1. `POST /invocations` com as 15 variáveis + JSON (até ~20 MiB no body).
2. Resposta **202** com `request_id` (`accept_latency_ms`).
3. O App grava o JSON no **mesmo** Volume `request_payloads`, pontua o Champion **no processo** (não chama Model Serving) e insere `inference_observability`.
4. `GET /invocations/{id}` até `done`. `total_latency_ms` cobre até o resultado.

Apps exigem **OAuth** (PAT redireciona para login). Jobs usam M2M (`validator-client-id` / `validator-client-secret`).

**B2 — Pré-processamento tabular (padrão clássico de prod, não empacotado como App)**

1. Consumidor (ou job) grava JSON no Volume e extrai features.
2. Endpoint recebe só o vetor tabular.
3. Observabilidade guarda `payload_uri` + features + predição.

- Prós B1: o cliente manda o JSON de uma vez; Serving não vê 15 MiB no HTTP.
- Prós B2: endpoint clássico, sem I/O no serving.
- Contras: feature parity treino/serving (`derive_features_from_json`); App tem fila in-memory.

#### C — Batch / assíncrono (`06_async_batch`)

1. Enfileira em `async_inference_queue`.
2. Job scoreia `@Champion`.
3. Resultado volta na fila + observabilidade.

- Prós: sem hard limit de request online; barato para documentos grandes.
- Contras: não atende SLA síncrono de milissegundos/segundos.

## Observabilidade customizada

Tabela `inference_observability`:

- `request_id`, `event_ts`, `endpoint_name`, `model_version` / alias
- `status`, `accept_latency_ms` (envio + `request_id`), `total_latency_ms` (até o resultado), `latency_ms` (alias de total)
- `payload_uri`, `payload_hash`, `payload_bytes`
- `prediction`, `probability`, `error_message`

Jobs posteriores calculam drift, taxa de erro, latência p95 e custo.

### Retenção e dados sensíveis

- Prefira **não** reter o JSON completo se não for necessário para auditoria.
- Alternativas: campos mascarados, amostra, só hash, TTL curto no Volume.
- Inference tables continuam úteis para requests pequenos; aqui tratam-se como complemento, não store primário.

## Rollback

1. Repontuar alias `Champion` para a versão anterior.
2. Rodar `deploy_endpoint` (update-config) para a versão anterior.
3. Poll `state.ready == READY` **e** `state.config_update == NOT_UPDATING`.

## Checklist de produção

- [ ] Catálogos e grants por grupo/SP (sem user individual em prod)
- [ ] Bundle deployado por CI, não por notebook manual
- [ ] Gates de promoção com métricas + signature + smoke test
- [ ] Decisão A/B/C documentada para o JSON de 16 MB
- [ ] Volume + observability com política de retenção/PII
- [ ] Alertas de erro/latência/drift
- [ ] Runbook de rollback
