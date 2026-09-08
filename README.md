# Multimodal Surgical Video QA

Ask natural-language questions about a surgical video and get answers grounded
in what the video actually shows and says.

The pipeline detects and segments instruments frame by frame, transcribes the
audio, folds the detections into a temporal knowledge graph, and answers
questions by routing each one to graph traversal, vector search over the
narration, or both.

```
video ──┬─► frame sampling (OpenCV) ──► YOLO detection ──► box-prompted SAM
        │                                     │                    │
        │                                     └──── intervals ─────┘
        │                                              │
        └─► Whisper transcription ──► segments ─────────┤
                                                        ▼
                                        Neo4j knowledge graph + Chroma index
                                                        │
                              question ──► router ──────┤
                                                        ▼
                                        graph / vector / hybrid retrieval
                                                        │
                                                        ▼
                                            grounded, cited answer
```

## Run it right now

No GPU, no weights, no database, no API key:

```bash
pip install fastapi uvicorn pydantic pydantic-settings opencv-python-headless numpy pytest PyYAML httpx
make demo        # generates a sample video, ingests it, asks six questions
make test        # 89 tests
```

`SVQA_BACKEND=stub` swaps every heavy model for a deterministic fake — the
whole pipeline, API and test suite run on a laptop in about two seconds. The
stub is not a mock of the interfaces; it is a real implementation of them, so
the code paths under test are the same ones that run in production.

Serve the API:

```bash
make run                                   # http://localhost:8000/docs
curl -X POST localhost:8000/videos/ingest -H 'content-type: application/json' \
     -d '{"path":"data/raw/procedure.mp4","sample_fps":2}'
curl -X POST localhost:8000/ask -H 'content-type: application/json' \
     -d '{"question":"when was the clipper used?","video_id":"procedure-a1b2c3d4"}'
```

## The interesting parts

### Detections become intervals, not nodes

A 40-minute procedure at 2 fps with two instruments per frame is ~9,600
detections. Written naively that is 9,600 graph nodes that answer nothing
useful, because "when was the hook in use" becomes a scan over the whole video.

`graph/events.py` folds contiguous detections into `Event` intervals — roughly
20-80 nodes per video, and the same question becomes one indexed range query.
The details that matter are the thresholds: a detector that drops a single
frame mid-use must not split one use into two events, and a one-frame flicker
at low confidence must not become an event at all. Both are configurable and
both are covered in `tests/test_events.py`.

### Hybrid retrieval, routed by rules first

Graph answers *when* and *how many*. Vector search answers *what was said about
it*. Most interesting questions need one or the other, and some need both.

An LLM router works but costs a call and a few hundred milliseconds on every
question, and it is non-deterministic on exactly the questions users repeat
most. So `retrieval/planner.py` routes on signal words and extracted entities
first — instrument names, time windows like "between 2 and 4 minutes", phase
names — and resolves most real questions deterministically, with an LLM
consulted only when the rules are genuinely ambiguous.

The subtle case, and the one that caught a real bug during development:

> "What was said while the clipper was out?"

This reads as pure narration — the only obvious signal is "said". But the
"while" clause is a time constraint that only the graph can resolve, and
routing it to vector-only silently loses the window. Temporal conjunctions are
therefore graph signals, which sends this question down the hybrid path where
`MENTIONED_DURING` edges join the two modalities in one traversal.

### Generated Cypher is treated as untrusted

Text-to-Cypher hands an LLM a database connection, and the prompt reaching it
may contain text lifted from a transcript. `graph/guard.py` validates generated
queries in code before execution — single statement, read-only opener, no write
or admin clause outside a string literal, mandatory LIMIT — and the Neo4j
session is additionally opened in `READ` access mode against a least-privilege
user. The session mode is the real control; the validator gives a clear error
and a cheap audit log. Templates are always preferred over generation.

### Dynamic model loading with a bounded resident set

YOLOv8n is ~6 MB. SAM ViT-H is ~2.4 GB. Loading every variant at startup blows
the container memory limit; loading per request wastes seconds of GPU transfer.
`vision/registry.py` registers models as loader callables, materialises them on
first use, and evicts least-recently-used past capacity.

Two details that only show up under concurrent load:

- Loading holds a **per-key** lock, not a global one, so ten requests for the
  same cold model trigger one load while a request for a different model is not
  queued behind it.
- The cache is mutated only after a successful load, so a loader that raises on
  missing weights leaves no poisoned entry to serve later.

Both are tested directly, including the ten-threads-one-cold-model race.

### Benchmarking, because the SAM choice is a real trade-off

MobileSAM vs FastSAM vs full SAM is roughly a 20x latency spread against
noticeably different mask quality, and the ranking changes completely between
CPU and GPU. `bench/benchmark.py` discards warmup frames (the first inference
pays CUDA context creation and cuDNN autotuning) and reports p95 alongside the
mean, because the tail is set by the frames where the detector found eight
boxes and the segmenter had to decode eight masks.

```bash
PYTHONPATH=src python scripts/bench.py data/raw/procedure.mp4 --frames 64
```

## Graph schema

```
(:Video)-[:HAS_EVENT]->(:Event)-[:OF_INSTRUMENT]->(:Instrument)
(:Video)-[:HAS_PHASE]->(:Phase)      (:Event)-[:DURING]->(:Phase)
(:Video)-[:HAS_SEGMENT]->(:Segment)  (:Event)-[:MENTIONED_DURING {overlap_s}]->(:Segment)
(:Event)-[:PRECEDES {gap_s}]->(:Event)
```

`Instrument` is its own node so cross-video questions are one hop.
`PRECEDES` is denormalised on write because "what happened immediately after
the clipper" is what people ask, and an explicit edge makes it a traversal
rather than a sort. Phases are rule-based (`configs/phases.yaml`) rather than a
learned classifier: no phase-labelled training data required, and inspectable,
which matters when someone asks why the graph says "dissection".

## Real mode

```bash
cp .env.example .env          # set SVQA_BACKEND=real, add SVQA_GEMINI_API_KEY
docker compose up --build     # Neo4j + API
```

Weights go in `data/artifacts/`:

| file | source |
|---|---|
| `yolov8n-surgical.pt` | train it: `python scripts/train_yolo.py --data dataset/data.yaml` |
| `mobile_sam.pt` | MobileSAM repo |
| `sam_vit_h_4b8939.pth` | Meta SAM release |
| `FastSAM-s.pt` | downloaded automatically by ultralytics |

For training data, Roboflow Universe has public surgical-instrument detection
sets; Cholec80 and m2cai16-tool are the standard academic ones but need an
access request. Generate a matching `data.yaml` with
`python scripts/train_yolo.py --make-data-yaml dataset/` — class order must
match `INSTRUMENT_CLASSES` or every prediction is silently mislabelled.

## Deployment

```bash
make k8s-local     # kind cluster, build, load, apply, wait for rollout
```

`k8s/` runs two replicas behind a ClusterIP service with a CPU-target HPA.
Three choices worth flagging:

- **One uvicorn worker per pod.** The registry is per-process, so N workers
  means N copies of every resident checkpoint. Scale with replicas.
- **A startup probe carries cold start** so liveness can stay tight. Without
  it, a slow first model load looks like a hang and the kubelet restarts the
  pod in a loop.
- **HPA scales up two pods at a time**, not by doubling, because each new pod
  pays a cold model load and a stampede hits checkpoint storage together.

The vector index is on an `emptyDir` here, which is fine for a demo and wrong
for anything real — two replicas get two separate indexes. A PVC or a managed
vector service is the fix.

## Layout

```
src/svqa/
  vision/      registry (LRU, lazy), YOLO detector, SAM segmenter, OpenCV frames
  audio/       Whisper transcription
  graph/       event derivation, Neo4j schema + client, read-only Cypher guard
  retrieval/   query planner, vector store, LLM adapters, ask engine
  pipeline/    end-to-end ingest with per-stage timing
  bench/       latency harness
  api/         FastAPI app, routers, schemas
tests/         89 tests, all runnable with no GPU/database/API key
k8s/           deployment, service, HPA, config
scripts/       train_yolo.py, demo.py, bench.py
```

## What this does not do

Being explicit, because a project page that claims everything is worth less
than one that draws the line:

- **No clinical validity.** Metrics are on public datasets with no clinical
  validation. The prompt forbids the model from offering surgical judgement,
  and the answer layer reports what the evidence shows rather than assessing
  technique.
- **Phase inference is rules, not a model.** It has no notion of a phase that
  is defined by anatomy rather than by which instruments are visible.
- **Tracking is per-frame, not multi-object tracking.** Two graspers in one
  frame are two detections of "grasper", not two tracked instances. Adding
  ByteTrack or BoT-SORT for identity would be the next real step.
- **Video is not multimodally embedded.** "Multimodal" here means vision and
  audio are fused in the *graph*, joined on time. A video-native embedding
  model (or frame captioning into the vector index) would let the vector path
  answer visual questions, which today only the graph can.
- **The stub vector index is lexical, not semantic.** TF-IDF cosine with crude
  stemming, so it matches "clipper" to "clipping" but has no real notion of
  meaning. Real mode uses sentence-transformers embeddings in Chroma.
- **Ingest is synchronous.** Fine while the pipeline is being tuned, since the
  response carries per-stage timings. Production needs a task queue.

## License

MIT.
