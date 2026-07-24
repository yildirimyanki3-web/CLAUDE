"""Unit tests for dnlssm.reports.report_builder."""

from __future__ import annotations

import pandas as pd

from dnlssm.config import load_experiment_config
from dnlssm.data.provenance import ProvenanceLog, ProvenanceRecord
from dnlssm.reports.report_builder import ReportContext, build_experiment_report


class TestBuildExperimentReportMinimal:
    def test_minimal_context_states_stages_not_run(self) -> None:
        config = load_experiment_config()
        context = ReportContext(experiment_id="test_exp_1", config=config)
        report = build_experiment_report(context)
        assert "test_exp_1" in report
        assert "was not run in this experiment" in report
        assert "No economic" in report

    def test_report_never_contains_banned_narrative_terms(self) -> None:
        config = load_experiment_config()
        context = ReportContext(experiment_id="test_exp", config=config)
        report = build_experiment_report(context)
        for banned in ("regime", "boom", "crisis", "financial repression"):
            assert banned not in report.lower()


class TestBuildExperimentReportWithProvenance:
    def test_provenance_section_reports_unresolved(self) -> None:
        config = load_experiment_config()
        log = ProvenanceLog()
        log.add(ProvenanceRecord(canonical_id="policy_rate", status="failed_all_providers"))
        log.add(ProvenanceRecord(canonical_id="usdtry_fx", status="success", resolved_provider="fred"))
        context = ReportContext(experiment_id="test_exp", config=config, provenance=log)
        report = build_experiment_report(context)
        assert "policy_rate" in report
        assert "Unresolved variables" in report


class TestBuildExperimentReportSections:
    def test_model_specification_section_present(self) -> None:
        config = load_experiment_config()
        context = ReportContext(experiment_id="test_exp", config=config)
        report = build_experiment_report(context)
        assert "Model Specification" in report
        assert config.model.transition_function.family in report

    def test_residual_stats_table_rendered(self) -> None:
        config = load_experiment_config()
        stats = pd.DataFrame({"variable": ["a", "b"], "rmse": [0.1, 0.2], "mae": [0.05, 0.15]})
        context = ReportContext(experiment_id="test_exp", config=config, residual_stats=stats)
        report = build_experiment_report(context)
        assert "rmse" in report
        assert "0.1000" in report or "0.1" in report

    def test_reproducibility_section_has_seed(self) -> None:
        config = load_experiment_config()
        context = ReportContext(experiment_id="test_exp", config=config)
        report = build_experiment_report(context)
        assert str(config.runtime.global_seed) in report
