import pytest


def test_health_reports_index_and_resources(client):
    payload = client.get("/health").json()
    assert payload["status"] in {"ok", "degraded"}
    assert payload["index"]["loaded"] is True
    assert payload["index"]["chunks"] > 0
    assert "gpus" in payload["resources"]


def test_metrics_exposes_prometheus_histograms(client):
    client.post("/ask", json={"question": "How much VRAM does an RTX 4060 have?"})
    body = client.get("/metrics").text
    assert "voicerag_stage_latency_seconds" in body
    assert "voicerag_requests_total" in body
    assert "voicerag_process_rss_bytes" in body


def test_ask_returns_answer_sources_and_request_id(client):
    response = client.post("/ask", json={"question": "How much VRAM does an RTX 4060 have?"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["sources"]
    assert payload["sources"][0]["source"] == "04_gpu_capacity.md"
    assert payload["prompt"]["ref"].startswith("rag_answer@")
    assert payload["request_id"] == response.headers["X-Request-ID"]


def test_ask_accepts_a_pinned_prompt_version(client):
    payload = client.post(
        "/ask", json={"question": "gpu memory", "prompt_ref": "rag_answer@v2"}
    ).json()
    assert payload["prompt"]["ref"] == "rag_answer@v2"


def test_ask_rejects_an_unknown_prompt_version(client):
    response = client.post("/ask", json={"question": "gpu", "prompt_ref": "rag_answer@v99"})
    assert response.status_code == 400
    assert "v99" in response.json()["detail"]


@pytest.mark.parametrize("payload", [{}, {"question": "x"}, {"question": "ok", "top_n": 0}])
def test_ask_validates_input(client, payload):
    assert client.post("/ask", json=payload).status_code == 422


def test_search_returns_candidates_and_timings(client):
    payload = client.post("/search", json={"query": "reranking", "top_n": 3}).json()
    assert len(payload["chunks"]) == 3
    assert payload["candidates_considered"] >= 3
    assert {"embed", "search", "rerank"} <= set(payload["timings_ms"])


def test_prompts_endpoint_lists_every_version_with_hashes(client):
    rows = client.get("/prompts").json()
    refs = {row["ref"] for row in rows}
    assert {"rag_answer@v1", "rag_answer@v2", "rag_answer@v3"} <= refs
    assert all(len(row["sha256"]) == 16 for row in rows)


def test_prompts_diff_endpoint(client):
    body = client.get("/prompts/diff", params={"a": "rag_answer@v1", "b": "rag_answer@v3"}).text
    assert "rag_answer@v1" in body and "rag_answer@v3" in body


def test_transcribe_returns_quality_signals(client, wav_file):
    with wav_file.open("rb") as fh:
        response = client.post("/transcribe", files={"file": ("sample.wav", fh, "audio/wav")})
    payload = response.json()
    assert response.status_code == 200
    assert payload["text"]
    assert payload["duration_s"] == pytest.approx(1.0, abs=0.05)
    assert payload["real_time_factor"] >= 0


def test_voice_ask_transcribes_then_answers(client, wav_file):
    with wav_file.open("rb") as fh:
        response = client.post("/voice-ask", files={"file": ("sample.wav", fh, "audio/wav")})
    payload = response.json()
    assert response.status_code == 200
    assert payload["transcript"]["text"]
    assert payload["answer"]
    assert payload["sources"]


def test_voice_ask_rejects_unsupported_format(client, wav_file):
    with wav_file.open("rb") as fh:
        response = client.post("/voice-ask", files={"file": ("notes.txt", fh, "text/plain")})
    assert response.status_code == 415


def test_voice_ask_rejects_empty_upload(client):
    response = client.post("/voice-ask", files={"file": ("empty.wav", b"", "audio/wav")})
    assert response.status_code == 400


def test_openapi_schema_is_served(client):
    schema = client.get("/openapi.json").json()
    assert "/voice-ask" in schema["paths"]
    assert "/metrics" in schema["paths"]
