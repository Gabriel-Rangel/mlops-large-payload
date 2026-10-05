# Resposta sugerida ao cliente (pronta para envio)

Olá — obrigado pelo contexto. Segue uma proposta objetiva para sair do notebook de POC e chegar a um fluxo de MLOps em produção, já considerando o JSON de ~16 MB.

## 1) Governança e ciclo de vida dos modelos

Recomendamos evoluir o notebook para um fluxo automatizado, separando **código/infra** de **governança do modelo**:

**Código e infraestrutura**

- Versionar treino, avaliação, registro e deploy em Git.
- Usar **Databricks Asset Bundles** (e Terraform onde fizer sentido em account-level) para promover Jobs, endpoints, permissões e configuração entre ambientes.
- Evitar alterações manuais em produção.

**Governança no Unity Catalog (MLflow)**

- Registrar modelos no UC.
- Organizar **catálogos separados** (`dev` / `staging` / `prod`).
- Permissões por **grupos e service principals** (não usuários individuais).
- Usar aliases **`Challenger`** (candidato) e **`Champion`** (versão ativa no endpoint).
- Gates antes da promoção: métricas, schema/signature, testes funcionais, performance, segurança/compliance.
- Registrar datasets de treino/avaliação no MLflow para preservar lineage dados↔modelo.

Fluxo alvo:

```
Git → CI (lint/testes) → treino (dev) → registro UC → avaliação
  → promoção staging → testes de integração/carga → aprovação
  → promoção prod → alias Champion → update do Model Serving
  → monitoramento e rollback
```

O Unity Catalog cobre acesso, auditoria, lineage e descoberta; a promoção pode ser controlada por permissões de leitura em staging e escrita em prod, complementada pelos aliases.

## 2) Payloads > 1 MiB e o JSON de ~16 MB

Há **duas** limitações distintas:

1. **Custom Model Serving** aceita até ~**16 MB** por requisição. Um campo JSON de ~16 MB + as outras 15 variáveis + serialização fica **no limite** e pode falhar.
2. Mesmo que a inferência passe, payloads **> ~1 MiB** **não** são registrados nas **inference tables**.

### Recomendação: não enviar o JSON completo no request

1. Gravar o JSON em um **Unity Catalog Volume** (ou Delta, se preferirem tabular).
2. Gerar `request_id` + `payload_uri` (+ `payload_hash`).
3. Enviar ao endpoint apenas as **15 variáveis** + a **referência**.
4. O modelo (ou uma camada de pré-processamento) recupera o JSON e processa.
5. A observabilidade guarda metadados e o ponteiro — **não** o JSON completo.

Exemplo de body:

```json
{
  "var_1": "...",
  "var_15": "...",
  "payload_uri": "/Volumes/prod.ml.payloads/requests/abc123.json",
  "payload_hash": "sha256:..."
}
```

O identity do Model Serving (ou da camada de pré-processamento) precisa de **leitura** no Volume.

### Se o JSON “precisar” ir no request

- **Reduzir**: remover campos não usados, compactar/normalizar, enviar só trechos relevantes; medir o tamanho **real** com o envelope.
- **Particionar** com `request_id` comum e persistir partes fora do endpoint (não manter estado em memória no serving).
- **Migrar para batch/assíncrono**: gravar entradas em Delta, Job/pipeline scoreia, grava predições — em geral melhor quando não há SLA síncrono de poucos segundos.

### Logging

Mesmo com telemetria customizada existem limites (não embutir 16 MB em linhas de log). Guardem **referência + hash + metadados**; avaliem retenção e PII — muitas vezes o payload completo não precisa ser armazenado.

As inference tables continuam úteis para payloads pequenos; para o caso de vocês, tratem-nas como complemento. A tabela primária de monitoramento deve ser **customizada** (métricas + ponteiro para o Volume).

## 3) Próximos passos práticos

1. Definir estratégia **dev / staging / prod** (catálogos e workspaces).
2. Transformar o notebook em **Jobs parametrizados**.
3. Criar repositório Git (treino, avaliação, registro, deploy).
4. Implementar CI/CD com **Databricks Asset Bundles**.
5. Definir grupos, service principals, owners e permissões de produção.
6. Padronizar tags do modelo (owner, domínio, criticidade, versão de código, dataset, SLA, status de validação).
7. Criar **Volume + tabela de observabilidade** para payloads > 1 MiB.
8. Definir métricas (latência, erro, volume, drift, qualidade, custo) e política de **rollback / retenção**.

Montamos um projeto de referência (`mlops-large-payload`) com esse desenho — Bundle, Jobs, PyFunc com `payload_uri` (Serving lê o Volume via Files API), Databricks App assíncrona que recebe o JSON e pontua localmente, tabela `inference_observability` (metadados + `accept_latency_ms` / `total_latency_ms`) e notebooks de validação/curl — para usarmos como base no workshop/hands-on.

Fico à disposição para alinharmos se a inferência precisa ser **síncrona** (Serving + referência) ou se **async** (App 202 + poll) atende o use case.
