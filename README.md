# WebArena Data Pipeline — Team 4 (DATA 298A)

**MSDA Project I | San Jose State University**  
**Team:** Alisha Kartik, Ankita Rawat, Kapil Reddy Sanikommu

---

## What This Project Does

This repository contains a fully automated data collection and processing pipeline for the **WebArena-Verified** benchmark. The pipeline collects web interaction trajectories from a simulated e-commerce shopping site using a rule-based Playwright agent, processes them, and stores them in Databricks for training AI models that can navigate websites autonomously.

The end goal is to train an RL-based agent that can take natural language instructions like *"Find a laptop bag and add it to the cart"* and complete them on a real website.

---

## Pipeline Architecture

```
WebArena-Verified Shopping Site (Docker :7770)
            ↓
    collect_data.py          ← Playwright agent browses site, saves successful episodes
            ↓
    export_raw_dataset.py    ← Merges all episodes into raw_dataset_success_only.jsonl
            ↓
    preprocess_data.py       ← Cleans, validates, standardizes records
            ↓
    transform_data.py        ← Converts to SFT format (input/output pairs)
            ↓
    split_data.py            ← Splits 70% train / 15% val / 15% test
            ↓
    databricks_uploader.py   ← Uploads to Databricks Unity Catalog Volume
```

All steps are orchestrated by an **Apache Airflow DAG** running in Docker, scheduled daily at 2 AM UTC.

---

## Tech Stack

| Tool | Purpose |
|---|---|
| WebArena-Verified | Simulated Magento shopping site for agent interaction |
| Playwright | Headless browser automation for data collection |
| Apache Airflow | Pipeline orchestration and scheduling |
| Docker | Containerization of all services |
| Databricks | Cloud storage (Unity Catalog Volumes) |
| Python 3.11 | All pipeline scripts |

---

## Data Statistics

| Stage | Count |
|---|---|
| Episodes collected | 520 |
| Total steps | 1,212 |
| Preprocessed | 1,212 |
| Transformed samples | 1,212 |
| Train (70%) | 848 |
| Validation (15%) | 182 |
| Test (15%) | 182 |
| Task types | 3 (search, add-to-cart, navigation) |
| Unique instructions | 29 |

---

## Data Format

### Raw Episode Sample
```json
{
  "instruction": "Search for a bag",
  "observation_before": {
    "url": "http://localhost:7770",
    "visible_buttons": ["Search", "Add to Cart"],
    "visible_inputs": [{"name": "q", "type": "text"}],
    "search_box_present": true
  },
  "action": {"type": "TYPE", "target": "input[name=\"q\"]", "value": "bag"},
  "reward": 1,
  "done": true
}
```

### Transformed SFT Sample
```json
{
  "input": "Instruction: Search for a bag\n\nVisible Buttons: Search, Add to Cart\n\nInput Fields: q",
  "output": "TYPE input[name=\"q\"] bag"
}
```

---

## Prerequisites

- **Docker Desktop** installed and running
- **Python 3.11+**
- **20 GB free disk space** (WebArena Docker images are ~5GB each)
- Databricks workspace access (ask Kapil for token)

---

## Setup Instructions

### Step 1 — Clone the repo
```bash
git clone https://github.com/ankitarawat1310/webarena-agent-pipeline.git
cd webarena-agent-pipeline
git checkout airflow-pipeline
```

### Step 2 — Create .env file
Create a `.env` file in the root directory:
```
DATABRICKS_HOST=https://dbc-fc7d5db5-8e46.cloud.databricks.com
DATABRICKS_TOKEN=<ask Kapil for token>
NUM_TASKS_PER_SITE=10
MAX_STEPS_PER_TASK=8
```

### Step 3 — Pull WebArena Docker images
```bash
docker pull am1n3e/webarena-verified-shopping
docker pull am1n3e/webarena-verified-shopping_admin
```
> This takes 20-40 minutes — images are ~5GB each.

### Step 4 — Initialize Airflow (first time only)
```bash
docker compose up airflow-init
```
Wait for: `airflow-init exited with code 0`

### Step 5 — Start all services
```bash
docker compose up -d
```
Wait 2-3 minutes for everything to start.

### Step 6 — Verify services are running
```bash
docker compose ps
```

### Step 7 — Open Airflow UI
Go to: http://localhost:8081
Username: `admin` | Password: `admin`

---

## Running the Pipeline

### Option 1 — Full automated run via Airflow
1. Go to http://localhost:8081
2. Enable the `webarena_verified_pipeline` DAG
3. Click **Trigger DAG**
4. Watch all tasks turn green (~4-5 minutes)

### Option 2 — Collect data then trigger pipeline

Collect specific number of new episodes:
```bash
python scripts/collect_data.py --add 10
```

Then trigger Airflow DAG to process and upload.

### Option 3 — Run scripts manually
```bash
python scripts/collect_data.py --add 10
python scripts/export_raw_dataset.py
python scripts/preprocess_data.py
python scripts/transform_data.py
python scripts/split_data.py
python scripts/databricks_uploader.py --run-id 20260409 --data-root data --volume-root /Volumes/workspace/webarena/trajectories
```

---

## Project Structure

```
webarena-agent-pipeline/
├── docker-compose.yml              ← All Docker services
├── .env.example                    ← Copy to .env and fill credentials
├── README.md                       ← This file
├── dags/
│   └── webarena_pipeline_dag.py    ← Airflow DAG definition
├── scripts/
│   ├── collect_data.py             ← Agent data collection (--add N flag)
│   ├── export_raw_dataset.py       ← Merge raw episodes
│   ├── preprocess_data.py          ← Clean and validate
│   ├── transform_data.py           ← Convert to SFT format
│   ├── split_data.py               ← Train/val/test split
│   ├── databricks_uploader.py      ← Upload to Databricks
│   └── test.py                     ← Count trajectories
├── src/
│   ├── agent/
│   │   ├── baseline_agent.py       ← Rule-based agent logic
│   │   └── action_schema.py        ← Action execution (CLICK, TYPE, STOP)
│   ├── data/
│   │   ├── preprocess.py           ← Preprocessing logic
│   │   ├── transform.py            ← Transformation logic
│   │   └── logger.py               ← Episode saving
│   └── env/
│       └── browser_env.py          ← Page observation
└── config/
    └── tasks.yaml                  ← Task definitions
```

---

## Airflow DAG Tasks

```
collect_data → export_raw_dataset → preprocess_data → transform_data → split_data → upload_to_databricks → pipeline_summary
```

---

## Services and Ports

| Service | URL | Description |
|---|---|---|
| Airflow UI | http://localhost:8081 | Pipeline monitoring (admin/admin) |
| WebArena Shopping | http://localhost:7770 | Shopping site the agent browses |
| WebArena Admin | http://localhost:7780 | Admin panel |

---

## Troubleshooting

**Port 8081 already in use**
→ Change `8081:8080` to `8082:8080` in `docker-compose.yml`

**Airflow not loading after startup**
→ Wait 2-3 more minutes, it is slow to start

**Collection task fails**
→ Run `docker compose ps` and wait for WebArena sites to show healthy

**Databricks upload fails**
→ Check `.env` has correct `DATABRICKS_HOST` and token starting with `dapi`

**Playwright not found error**
→ Run: `docker compose restart webarena-collector`

---

## Databricks Storage

Data is stored in Unity Catalog Volumes organized by run date:
```
/Volumes/workspace/webarena/trajectories/
    20260409/
        raw/         ← raw_dataset_success_only.jsonl
        processed/   ← processed_data.jsonl
        transformed/ ← dataset.jsonl
        splits/      ← train.jsonl, val.jsonl, test.jsonl
```

---

## Notes

- The `--add N` flag in `collect_data.py` automatically counts existing episodes and adds N more
- The pipeline overwrites processed/transformed/split files on each run
- Daily schedule runs at 2 AM UTC automatically when Docker is running
- For Databricks token access, contact Kapil

---

## References

1. Zhou et al. (2023). WebArena: A Realistic Web Environment for Building Autonomous Agents. https://arxiv.org/abs/2307.13854
2. WebArena-Verified. https://github.com/ServiceNow/webarena-verified
3. Apache Airflow Documentation. https://airflow.apache.org/docs/
4. Docker Documentation. https://docs.docker.com/
5. Playwright Python Documentation. https://playwright.dev/python/docs/intro
6. Databricks Unity Catalog. https://docs.databricks.com/en/data-governance/unity-catalog/index.html
