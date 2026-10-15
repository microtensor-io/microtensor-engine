# microtensor-engine

A Python SDK for running Microtensor certified specialists: hosted through the API today, inside your own environment with the Enterprise preview.

The hosted API needs no SDK; any OpenAI client works. The Engine is the option for private deployment and for applications that want task to specialist mapping, a quality floor and escalation handled in one object.

## Interface

    from microtensor import Engine

    Engine.frontier_models()   # live list of frontier models our inference miners serve

    engine = Engine.load(
        tasks=["text-to-sql", "classification", "extraction"],
        device="cuda",
        quality_floor=0.9,
        escalation=True,                    # False = never escalate, return flagged answers
        escalation_model="auto",            # or a model id from frontier_models(), or a dict per task
        escalation_budget={"usd_per_day": 50},
    )

    result = engine.run("Which customers churned last quarter?", context=schema)
    # result.answer, result.task, result.confidence, result.escalated,
    # result.escalated_to, result.below_floor, result.cost

## Behaviour

- `tasks` maps each task to the current certified specialist in its arena. Users never see arenas, and get upgrades automatically when a better specialist is certified. A task the network does not serve yet raises `TaskUnavailable` at load time, naming what is served.
- The certified router identifies the task for each request; `result.task` says which specialist answered. Pass `task=` to `run` to skip classification.
- Escalation stays inside the Microtensor network, served by frontier models hosted by inference miners. `escalation_model` takes `"auto"`, a model id from `frontier_models()`, or a dict per task such as `{"text-to-sql": "deepseek-v4-pro", "classification": False}`.
- With escalation off, with no frontier model served, or once the daily budget is used, answers below the floor come back with `below_floor=True` and the application decides.
- `frontier_models()` reads the live catalogue, so a model the network does not serve cannot be asked for.

## Backends

| Backend | Status |
|---|---|
| `hosted` (default) | Calls the API with `MICROTENSOR_API_KEY`. Same answers, routing and confidence as the API. |
| `local` | Fetches the certified artifacts and runs them on your GPUs with the validators' runtime: vLLM for the model, a classifier process for the router, a gVisor sandbox for the harness. Enterprise preview; `Engine.load(..., backend="local")` names the preview until then. |

The local backend will install with `pip install "microtensor-engine[local]"` and needs one 24 GB GPU per held specialist, Docker with the NVIDIA toolkit and gVisor, and a licence token for the private artifact mirrors.

## Router performance

Certified routers can be served with CUDA graphs and fused kernels for RMSNorm, SwiGLU and RoPE. These modes stay off by default and are enabled per model only after a parity check shows outputs and confidences match the certified run within tolerance on the certification items.

## Import name

The package installs as `microtensor`, the customer facing name. The network tooling installs as `microtensor_subnet` (distribution `microtensor-subnet`), so both can live in one environment.

## Development

    pip install -e ".[dev]"
    ruff check . && mypy microtensor && pytest -q

Docs: https://www.microtensor.cloud/docs/integrations/enterprise/engine
