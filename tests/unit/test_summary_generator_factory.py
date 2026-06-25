from src.app.clients.summarizer_client import FakeSummaryGenerator, VLLMSummaryGenerator
from src.app.workers import insight_tasks


def test_build_fake_summarizer_when_enabled(monkeypatch):
    monkeypatch.setattr(insight_tasks.settings, "use_fake_summarizer", True)

    generator = insight_tasks.build_summary_generator()

    assert isinstance(generator, FakeSummaryGenerator)


def test_build_summary_generator_uses_vllm_when_disabled(monkeypatch):
    monkeypatch.setattr(insight_tasks.settings, "use_fake_summarizer", False)
    monkeypatch.setattr(insight_tasks.settings, "vllm_base_url", "http://localhost:8001/v1")
    monkeypatch.setattr(insight_tasks.settings, "vllm_api_key", "demo-key")
    monkeypatch.setattr(insight_tasks.settings, "summary_model_name", "summary-sft")

    generator = insight_tasks.build_summary_generator()

    assert isinstance(generator, VLLMSummaryGenerator)
