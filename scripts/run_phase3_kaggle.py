from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np

from aigc_detector.phase3.artifacts import atomic_json, validate_completion
from aigc_detector.phase3.r1 import write_job_summary
from aigc_detector.phase3.r2 import select_promotion_regime
from aigc_detector.phase3.r3 import select_promotion_setting
from aigc_detector.phase3.r4 import select_promotion_policy
from aigc_detector.phase3.r6 import select_candidate


STAGES = ("r1", "r2", "r3", "r4", "r5", "r6", "r7")
FINAL_LOCKED_JOB = "r7_locked_final"
VALID_STATUSES = {"planned", "submitted", "running", "completed", "failed", "skipped", "selected", "rejected"}


@dataclass(frozen=True)
class JobSpec:
    name: str
    stage: str
    kernel_suffix: str
    job_dir: str
    generator: str
    generator_job: str
    dependencies: tuple[str, ...] = ()
    common_contract: bool = True
    required: tuple[str, ...] = ()
    optional: bool = False
    selection_dataset: str | None = None


def job_specs(enable_so400m: bool = False) -> dict[str, JobSpec]:
    specs = [
        JobSpec("r1_dinov3", "r1", "r1-dinov3-vitl16", "kaggle/phase3/r1/dinov3_vitl16", "generate_phase3_r1_kernel_metadata.py", "dinov3_vitl16", required=("r1_summary.json",)),
        JobSpec("r1_siglip2_large", "r1", "r1-siglip2-large-256", "kaggle/phase3/r1/siglip2_large_256", "generate_phase3_r1_kernel_metadata.py", "siglip2_large_256", required=("r1_summary.json",)),
        JobSpec("r1_siglip2_so400m", "r1", "r1-siglip2-so400m-256", "kaggle/phase3/r1/siglip2_so400m_256", "generate_phase3_r1_kernel_metadata.py", "siglip2_so400m_256", required=("r1_summary.json",), optional=not enable_so400m),
        JobSpec("r2_single", "r2", "r2-25k-single", "kaggle/phase3/r2/r2_25k_single", "generate_phase3_r2_kernel_metadata.py", "r2_25k_single", ("selection_r1",), required=("r2_summary.json",)),
        JobSpec("r2_compound", "r2", "r2-25k-compound", "kaggle/phase3/r2/r2_25k_compound", "generate_phase3_r2_kernel_metadata.py", "r2_25k_compound", ("selection_r1",), required=("r2_summary.json",)),
        JobSpec("r2_promotion", "r2", "r2-100k-promotion", "kaggle/phase3/r2/r2_100k_promotion", "generate_phase3_r2_kernel_metadata.py", "r2_100k_promotion", ("selection_r2",), required=("r2_summary.json", "recommended_candidate.json"), selection_dataset="r2"),
        *[JobSpec(f"r3_{setting}", "r3", f"r3-{setting}-25k", f"kaggle/phase3/r3/r3_{setting}_25k", "generate_phase3_r3_kernel_metadata.py", f"r3_{setting}_25k", ("r2_promotion",), required=("r3_summary.json",)) for setting in ("baseline", "mild", "medium", "strong")],
        JobSpec("r3_promotion", "r3", "r3-100k-promotion", "kaggle/phase3/r3/r3_100k_promotion", "generate_phase3_r3_kernel_metadata.py", "r3_100k_promotion", ("selection_r3",), required=("r3_summary.json", "recommended_candidate.json"), selection_dataset="r3"),
        *[JobSpec(f"r4_{policy}", "r4", f"r4-{policy.replace('_','-')}-25k", f"kaggle/phase3/r4/r4_{policy}_25k", "generate_phase3_r4_kernel_metadata.py", f"r4_{policy}_25k", ("r3_promotion",), required=("bias_policy_summary.json",)) for policy in ("class_balanced", "source_balanced", "source_quality_matched")],
        JobSpec("r4_promotion", "r4", "r4-100k-promotion", "kaggle/phase3/r4/r4_100k_promotion", "generate_phase3_r4_kernel_metadata.py", "r4_100k_promotion", ("selection_r4",), required=("bias_policy_summary.json", "recommended_candidate.json"), selection_dataset="r4"),
        JobSpec("r5_low", "r5", "r5-low", "kaggle/phase3/r5/r5_low", "generate_phase3_r5_kernel_metadata.py", "r5_low", ("r4_promotion",), required=("expert_summary.json",)),
        JobSpec("r5_high", "r5", "r5-high", "kaggle/phase3/r5/r5_high", "generate_phase3_r5_kernel_metadata.py", "r5_high", ("r4_promotion",), required=("expert_summary.json",)),
        JobSpec("r5_ensemble", "r5", "r5-ensemble", "kaggle/phase3/r5/r5_ensemble", "generate_phase3_r5_kernel_metadata.py", "r5_ensemble", ("r5_low", "r5_high"), common_contract=False, required=("r5_summary.json", "recommended_candidate.json", "val_logits.npz")),
        *[JobSpec(f"r6_{mode}", "r6", f"r6-{mode.replace('_','-')}", f"kaggle/phase3/r6/r6_{mode}", "generate_phase3_r6_kernel_metadata.py", f"r6_{mode}", ("r4_promotion",), required=("r6_summary.json",)) for mode in ("global_only", "mean_patch", "topk_patch", "attention_pool")],
        JobSpec("r6_global_plus_local", "r6", "r6-global-plus-local", "kaggle/phase3/r6/r6_global_plus_local", "generate_phase3_r6_kernel_metadata.py", "r6_global_plus_local", ("selection_r6_local",), required=("r6_summary.json",), selection_dataset="r6_local"),
        JobSpec("r6_promotion", "r6", "r6-promotion", "kaggle/phase3/r6/r6_promotion", "generate_phase3_r6_kernel_metadata.py", "r6_promotion", ("selection_r6",), required=("r6_summary.json", "recommended_candidate.json"), selection_dataset="r6"),
        JobSpec("r7_search", "r7", "r7-search", "kaggle/phase3/r7/r7_search", "generate_phase3_r7_kernel_metadata.py", "r7_search", ("r3_promotion", "r4_promotion", "r5_ensemble", "r6_promotion"), common_contract=False,
                required=("model_diversity.csv", "ensemble_search.csv", "phase3_summary.json", "recommended_candidate.json", "locked_candidate.json", "val_logits.npz")),
    ]
    return {spec.name: spec for spec in specs}


class Registry:
    def __init__(self, path: Path):
        self.path = path
        if path.is_file(): self.data = json.loads(path.read_text(encoding="utf-8"))
        else: self.data = {"schema_version": 1, "updated_at": None, "jobs": {}, "selections": {}, "events": []}

    def save(self):
        self.data["updated_at"] = datetime.now(timezone.utc).isoformat(); atomic_json(self.path, self.data)

    def ensure(self, spec: JobSpec, slug: str, output: Path, config: dict):
        self.data["jobs"].setdefault(spec.name, {"stage": spec.stage, "logical_job_name": spec.name,
            "kernel_slug": slug, "kernel_version": None, "config": config, "status": "skipped" if spec.optional else "planned",
            "submitted_timestamp": None, "upstream_dependencies": list(spec.dependencies), "attached_kernel_sources": [],
            "output_download_location": str(output), "completion_artifact_path": None, "selected": False,
            "validation_metrics": None, "failure_reason": None, "remote_status": None, "last_command": None,
            "last_exit_code": None, "local_artifacts_valid": False})

    def update(self, name: str, **values):
        if "status" in values and values["status"] not in VALID_STATUSES: raise ValueError(values["status"])
        self.data["jobs"][name].update(values); self.save()


def parse_kaggle_status(text: str) -> str:
    value = text.lower()
    if any(token in value for token in ("error", "failed", "cancelled", "canceled")): return "failed"
    if any(token in value for token in ("complete", "completed")): return "complete"
    if any(token in value for token in ("running", "queued", "pending")): return "running"
    return "unknown"


class Controller:
    def __init__(self, repo: Path, config: dict, registry: Registry,
                 runner: Callable[..., subprocess.CompletedProcess] = subprocess.run, enable_so400m=False):
        self.repo, self.config, self.registry, self.runner = repo, config, registry, runner
        self.specs = job_specs(enable_so400m); self.username = config["kaggle_username"]
        configured_root = Path(config.get("artifacts_root", "artifacts/track5/phase3"))
        self.artifacts = configured_root if configured_root.is_absolute() else repo / configured_root
        for spec in self.specs.values():
            output = self.artifacts / spec.stage / spec.name
            cfg_path = repo / spec.job_dir / "config.json"
            cfg = json.loads(cfg_path.read_text()) if cfg_path.is_file() else {}
            self.registry.ensure(spec, f"{self.username}/track5-phase3-{spec.kernel_suffix}", output, cfg)
            if not spec.optional and self.registry.data["jobs"][spec.name]["status"] == "skipped":
                self.registry.data["jobs"][spec.name]["status"] = "planned"
        self.registry.save()
        self.hydrate_existing()

    def command(self, args: list[str], mutate=False, check=True):
        normalized = " ".join(args).replace("-", "_")
        if FINAL_LOCKED_JOB in normalized: raise RuntimeError("ABSOLUTE SAFETY: locked final-test job cannot be submitted by Phase-3 controller")
        result = self.runner(args, cwd=self.repo, text=True, capture_output=True, check=False)
        if check and result.returncode:
            raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(args)}\n{result.stdout}\n{result.stderr}")
        return result

    def scoped(self, stage: str | None, through: str | None) -> list[JobSpec]:
        if stage: allowed = {stage}
        elif through: allowed = set(STAGES[:STAGES.index(through)+1])
        else: allowed = set(STAGES)
        return [spec for spec in self.specs.values() if spec.stage in allowed]

    def preflight(self, stages: set[str]) -> list[dict]:
        required = {"source_dataset"}
        if stages & {"r1", "r2", "r3", "r4", "r5", "r6"}: required.add("training_dataset")
        if "r1" in stages: required |= {"dinov3_asset", "siglip2_large_asset"}
        elif stages & {"r2", "r3", "r4", "r5", "r6"}:
            selected = self.registry.data["selections"].get("r1", {}).get("job_name")
            if selected:
                required.add({"r1_dinov3":"dinov3_asset", "r1_siglip2_large":"siglip2_large_asset",
                              "r1_siglip2_so400m":"siglip2_so400m_asset"}[selected])
        rows = []
        for key in sorted(required):
            value = self.config.get(key); valid = isinstance(value, str) and "/" in value and "REPLACE" not in value
            rows.append({"asset": key, "reference": value, "valid": valid})
        if self.config.get("enable_so400m"):
            value = self.config.get("siglip2_so400m_asset"); rows.append({"asset":"siglip2_so400m_asset","reference":value,"valid":bool(value and "/" in value)})
        return rows

    def dependency_ready(self, dependency: str) -> bool:
        if dependency.startswith("selection_"): return self.registry.data["selections"].get(dependency.removeprefix("selection_"), {}).get("status") == "selected"
        return self.registry.data["jobs"].get(dependency, {}).get("status") in {"completed", "selected"}

    def ready(self, spec: JobSpec) -> bool:
        job = self.registry.data["jobs"][spec.name]
        return job["status"] == "planned" and not spec.optional and all(self.dependency_ready(dep) for dep in spec.dependencies)

    def selected_job(self, stage: str) -> str:
        selection = self.registry.data["selections"].get(stage)
        if not selection or selection.get("status") != "selected": raise ValueError(f"No validated {stage} selection")
        return selection["job_name"]

    def kernel_sources(self, spec: JobSpec) -> list[str]:
        if spec.stage == "r2": return [self.registry.data["jobs"][self.selected_job("r1")]["kernel_slug"]]
        if spec.stage == "r3": return [self.registry.data["jobs"]["r2_promotion"]["kernel_slug"]]
        if spec.stage == "r4": return [self.registry.data["jobs"]["r3_promotion"]["kernel_slug"]]
        if spec.name in {"r5_low", "r5_high"} or spec.stage == "r6": return [self.registry.data["jobs"]["r4_promotion"]["kernel_slug"]]
        if spec.name == "r5_ensemble": return [self.registry.data["jobs"][name]["kernel_slug"] for name in ("r5_low", "r5_high")]
        if spec.name == "r7_search":
            names = ["r3_promotion", "r4_promotion", "r5_low", "r5_high", "r5_ensemble", "r6_promotion"]
            if self.config.get("include_r1_r2_in_r7"): names += [self.selected_job("r1"), "r2_promotion"]
            return [self.registry.data["jobs"][name]["kernel_slug"] for name in names]
        return []

    def selected_model_asset(self) -> str:
        selected = self.selected_job("r1")
        key = {"r1_dinov3":"dinov3_asset", "r1_siglip2_large":"siglip2_large_asset",
               "r1_siglip2_so400m":"siglip2_so400m_asset"}[selected]
        return self.config[key]

    def generator_command(self, spec: JobSpec) -> list[str]:
        command = [sys.executable, str(self.repo / "scripts" / spec.generator), "--job", spec.generator_job,
                   "--job-dir", str(self.repo / spec.job_dir), "--source-dataset", self.config["source_dataset"]]
        if spec.stage != "r7" and spec.name != "r5_ensemble": command += ["--data-dataset", self.config["training_dataset"]]
        asset = self.selected_model_asset() if spec.stage not in {"r1", "r7"} else None
        if spec.stage == "r1":
            key = {"r1_dinov3":"dinov3_asset", "r1_siglip2_large":"siglip2_large_asset", "r1_siglip2_so400m":"siglip2_so400m_asset"}[spec.name]
            command += ["--model-dataset", self.config[key]]
        elif spec.stage in {"r2", "r3", "r4", "r5"} and spec.name != "r5_ensemble": command += ["--model-dataset", asset]
        elif spec.stage == "r6": command += ["--model-dataset", asset]
        sources = self.kernel_sources(spec)
        if spec.stage == "r2": command += ["--r1-kernel-source", sources[0]]
        elif spec.stage == "r3": command += ["--r2-kernel-source", sources[0]]
        elif spec.stage == "r4": command += ["--r3-kernel-source", sources[0]]
        elif spec.name in {"r5_low", "r5_high"}: command += ["--r4-kernel-source", sources[0]]
        elif spec.name == "r5_ensemble": command += ["--low-kernel-source", sources[0], "--high-kernel-source", sources[1]]
        elif spec.stage == "r6": command += ["--r4-kernel-source", sources[0]]
        elif spec.name == "r7_search":
            for source in sources: command += ["--candidate-kernel-source", source]
        if spec.selection_dataset:
            flag = "--selection-dataset" if spec.stage == "r6" else "--promotion-config-dataset"
            command += [flag, self.selection_dataset_slug(spec.selection_dataset)]
        return command

    def selection_dataset_slug(self, name): return f"{self.username}/track5-{name.replace('_','-')}-promotion-config"

    def submit(self, spec: JobSpec, force=False):
        job = self.registry.data["jobs"][spec.name]
        if spec.optional: return False
        if job["status"] in {"completed", "selected", "submitted", "running"} and not force: return False
        if not all(self.dependency_ready(dep) for dep in spec.dependencies): return False
        if job["status"] != "planned" and not force: return False
        generator = self.generator_command(spec)
        try: self.command(generator)
        except Exception as error:
            self.registry.update(spec.name, status="failed", failure_reason=str(error), last_command=generator, last_exit_code=1); raise
        sources = self.kernel_sources(spec)
        push = ["kaggle", "kernels", "push", "-p", str(self.repo / spec.job_dir)]
        try: result = self.command(push)
        except Exception as error:
            self.registry.update(spec.name, status="failed", failure_reason=str(error), last_command=push, last_exit_code=1); raise
        self.registry.update(spec.name, status="submitted", submitted_timestamp=datetime.now(timezone.utc).isoformat(),
                             attached_kernel_sources=sources, last_command=push, last_exit_code=result.returncode,
                             failure_reason=None)
        return True

    def refresh_status(self, spec: JobSpec):
        job = self.registry.data["jobs"][spec.name]
        if job["status"] not in {"submitted", "running"}: return job["status"]
        command = ["kaggle", "kernels", "status", job["kernel_slug"]]; result = self.command(command, check=False)
        if result.returncode:
            self.registry.update(spec.name, failure_reason=result.stderr, last_command=command, last_exit_code=result.returncode); return "unknown"
        remote = parse_kaggle_status(result.stdout + result.stderr)
        status = "failed" if remote == "failed" else "running"
        self.registry.update(spec.name, status=status, remote_status=remote, last_command=command, last_exit_code=0,
                             failure_reason="Kaggle job failed" if remote == "failed" else None)
        return remote

    def _find(self, root: Path, name: str) -> Path | None:
        return next(iter(root.rglob(name)), None) if root.exists() else None

    def validate_job(self, spec: JobSpec) -> tuple[bool, str | None, dict | None, Path | None]:
        root = Path(self.registry.data["jobs"][spec.name]["output_download_location"])
        try:
            metrics, completion = None, None
            if spec.common_contract:
                marker = self._find(root, "COMPLETED.json")
                if marker is None: raise ValueError("missing COMPLETED.json")
                contract = validate_completion(marker.parent); metrics, completion = contract["metrics"], marker
            for required in spec.required:
                artifact = self._find(root, required)
                if artifact is None: raise ValueError(f"missing required artifact {required}")
                if required.endswith("_summary.json") or required in {"phase3_summary.json", "recommended_candidate.json"}:
                    document = json.loads(artifact.read_text())
                    if document.get("selection_split") != "validation" or document.get("final_test_evaluated") is not False:
                        raise ValueError(f"invalid validation-only summary {required}")
                if required == "val_logits.npz":
                    with np.load(artifact) as arrays:
                        if not {"logits", "labels"} <= set(arrays.files): raise ValueError("invalid validation logits archive")
            if not spec.common_contract:
                recommended = self._find(root, "recommended_candidate.json")
                if recommended:
                    document = json.loads(recommended.read_text())
                    if document.get("selection_split") != "validation" or document.get("final_test_evaluated") is not False:
                        raise ValueError("non-validation recommendation")
                    if document.get("candidate") is None: raise ValueError("recommendation has no deployable candidate")
                completion = recommended or self._find(root, spec.required[0])
            if spec.name == "r7_search":
                from aigc_detector.phase3.r7_locked_test import validate_lock
                lock = self._find(root, "locked_candidate.json")
                validate_lock(json.loads(lock.read_text()))
            return True, None, metrics, completion
        except Exception as error: return False, str(error), None, None

    def download(self, spec: JobSpec):
        job = self.registry.data["jobs"][spec.name]
        if job.get("remote_status") != "complete": return False
        output = Path(job["output_download_location"]); output.mkdir(parents=True, exist_ok=True)
        command = ["kaggle", "kernels", "output", job["kernel_slug"], "-p", str(output)]
        result = self.command(command, check=False)
        if result.returncode:
            self.registry.update(spec.name, failure_reason=result.stderr, last_command=command, last_exit_code=result.returncode); return False
        valid, reason, metrics, completion = self.validate_job(spec)
        self.registry.update(spec.name, status="completed" if valid else "failed", validation_metrics=metrics,
                             completion_artifact_path=str(completion) if completion else None,
                             local_artifacts_valid=valid,
                             failure_reason=None if valid else f"Scientifically incomplete: {reason}", last_command=command, last_exit_code=0)
        return valid

    def confirm_existing(self, spec: JobSpec) -> bool:
        job = self.registry.data["jobs"][spec.name]
        if not job.get("local_artifacts_valid") or job["status"] == "completed": return False
        command = ["kaggle", "kernels", "status", job["kernel_slug"]]; result = self.command(command, check=False)
        remote = parse_kaggle_status((result.stdout or "") + (result.stderr or "")) if not result.returncode else "unknown"
        if remote == "complete":
            self.registry.update(spec.name, status="completed", remote_status="complete", failure_reason=None,
                                 last_command=command, last_exit_code=0); return True
        if remote == "failed": self.registry.update(spec.name, status="failed", remote_status="failed", failure_reason="Kaggle job failed")
        return False

    def _summary(self, job_name: str, filename: str) -> Path:
        path = self._find(Path(self.registry.data["jobs"][job_name]["output_download_location"]), filename)
        if path is None: raise FileNotFoundError(f"{job_name}: {filename}")
        return path

    def _record_selection(self, stage: str, selected: dict, job_names: list[str], artifact: Path):
        candidate_id = selected.get("candidate_id") or selected.get("selected_setting") or selected.get("selected_policy")
        chosen = None
        for name in job_names:
            documents = list(Path(self.registry.data["jobs"][name]["output_download_location"]).rglob("*.json"))
            if any(candidate_id and candidate_id in path.read_text(errors="ignore") for path in documents): chosen = name; break
        chosen = chosen or job_names[0]
        self.registry.data["selections"][stage] = {"status":"selected", "job_name":chosen, "candidate_id":candidate_id,
            "artifact":str(artifact), "selected_at":datetime.now(timezone.utc).isoformat(), "validation_only":True}
        for name in job_names: self.registry.data["jobs"][name]["selected"] = name == chosen
        self.registry.save()

    def _write_selection_dataset(self, name: str, document: dict, remote_mutation: bool):
        directory = self.artifacts / name / "promotion_config"; directory.mkdir(parents=True, exist_ok=True)
        atomic_json(directory / ("selection_config.json" if name.startswith("r6") else "promotion_config.json"), document)
        atomic_json(directory / "dataset-metadata.json", {"title":f"Track5 {name} selection", "id":self.selection_dataset_slug(name), "licenses":[{"name":"other"}]})
        if remote_mutation:
            result = self.command(["kaggle", "datasets", "create", "-p", str(directory), "--dir-mode", "zip", "-r", "skip"], check=False)
            if result.returncode and "already exists" not in (result.stderr or "").lower(): raise RuntimeError(result.stderr)
        return directory

    def reconcile(self, remote_mutation=False):
        selections = self.registry.data["selections"]
        all_r1 = ("r1_dinov3", "r1_siglip2_large", "r1_siglip2_so400m")
        required_r1 = [name for name in all_r1 if self.registry.data["jobs"][name]["status"] != "skipped"]
        r1_jobs = [name for name in required_r1 if self.registry.data["jobs"][name]["status"] == "completed"]
        if "r1" not in selections and required_r1 and len(r1_jobs) == len(required_r1):
            rows=[]
            for name in r1_jobs: rows.extend(json.loads(self._summary(name,"r1_summary.json").read_text())["results"])
            output=self.artifacts/"r1"; write_job_summary(rows,output,float(self.config.get("baseline_clean",.9681)))
            doc=json.loads((output/"recommended_candidate.json").read_text()); self._record_selection("r1",doc["candidate"],r1_jobs,output/"recommended_candidate.json")
        if "r2" not in selections and all(self.registry.data["jobs"][n]["status"]=="completed" for n in ("r2_single","r2_compound")):
            selected=select_promotion_regime(json.loads(self._summary("r2_single","r2_summary.json").read_text()),json.loads(self._summary("r2_compound","r2_summary.json").read_text()),float(self.config.get("baseline_clean",.9681)))
            directory=self._write_selection_dataset("r2",selected,remote_mutation); self._record_selection("r2",selected,["r2_single","r2_compound"],directory/"promotion_config.json")
        r3_jobs=[f"r3_{x}" for x in ("baseline","mild","medium","strong")]
        if "r3" not in selections and all(self.registry.data["jobs"][n]["status"]=="completed" for n in r3_jobs):
            docs=[json.loads(self._summary(n,"r3_summary.json").read_text()) for n in r3_jobs]; selected=select_promotion_setting(docs,float(self.config.get("baseline_clean",.9681)))
            directory=self._write_selection_dataset("r3",selected,remote_mutation); self._record_selection("r3",selected,r3_jobs,directory/"promotion_config.json")
        r4_jobs=[f"r4_{x}" for x in ("class_balanced","source_balanced","source_quality_matched")]
        if "r4" not in selections and all(self.registry.data["jobs"][n]["status"]=="completed" for n in r4_jobs):
            docs=[json.loads(self._summary(n,"bias_policy_summary.json").read_text()) for n in r4_jobs]; selected=select_promotion_policy(docs,float(self.config.get("baseline_clean",.9681)))
            directory=self._write_selection_dataset("r4",selected,remote_mutation); self._record_selection("r4",selected,r4_jobs,directory/"promotion_config.json")
        primitives=[f"r6_{x}" for x in ("global_only","mean_patch","topk_patch","attention_pool")]
        if "r6_local" not in selections and all(self.registry.data["jobs"][n]["status"]=="completed" for n in primitives):
            docs=[json.loads(self._summary(n,"r6_summary.json").read_text()) for n in primitives]; selected=select_candidate(docs,float(self.config.get("baseline_clean",.9681)),True)
            directory=self._write_selection_dataset("r6_local",selected,remote_mutation); self._record_selection("r6_local",selected,primitives,directory/"selection_config.json")
        all_r6=primitives+["r6_global_plus_local"]
        if "r6" not in selections and all(self.registry.data["jobs"][n]["status"]=="completed" for n in all_r6):
            docs=[json.loads(self._summary(n,"r6_summary.json").read_text()) for n in all_r6]; selected=select_candidate(docs,float(self.config.get("baseline_clean",.9681)),False)
            directory=self._write_selection_dataset("r6",selected,remote_mutation); self._record_selection("r6",selected,all_r6,directory/"selection_config.json")
        for stage, name in (("r2","r2_promotion"),("r3","r3_promotion"),("r4","r4_promotion"),
                            ("r5","r5_ensemble"),("r6","r6_promotion"),("r7","r7_search")):
            if self.registry.data["jobs"][name]["status"] == "completed":
                self.registry.data["jobs"][name]["selected"] = True
                key = f"{stage}_champion"
                if key not in selections:
                    selections[key] = {"status":"selected", "job_name":name,
                        "artifact":self.registry.data["jobs"][name]["completion_artifact_path"],
                        "selected_at":datetime.now(timezone.utc).isoformat(), "validation_only":True}
        self.registry.save()

    def hydrate_existing(self):
        """Reuse local work only after scientific artifact validation."""
        changed = False
        for spec in self.specs.values():
            job = self.registry.data["jobs"][spec.name]
            if job["status"] not in {"planned", "failed"}: continue
            valid, _, metrics, completion = self.validate_job(spec)
            if valid:
                job.update({"local_artifacts_valid":True, "validation_metrics":metrics,
                            "completion_artifact_path":str(completion), "failure_reason":None})
                if job.get("remote_status") == "complete": job["status"] = "completed"
                changed = True
        if changed: self.registry.save()

    def plan(self, specs: list[JobSpec]) -> str:
        lines=[]
        for stage in STAGES:
            selected=[spec for spec in specs if spec.stage==stage]
            if not selected: continue
            lines.append(f"{stage.upper()}:")
            for spec in selected:
                job=self.registry.data["jobs"][spec.name]
                state="READY" if self.ready(spec) else (job["status"].upper() if job["status"]!="planned" else "BLOCKED")
                deps=", ".join(spec.dependencies) or "none"
                lines.append(f"  {spec.name:30} {state:10} {job['kernel_slug']} deps=[{deps}]")
                if self.ready(spec): lines.append("    "+" ".join(self.generator_command(spec)))
        lines.append("FINAL TEST: PERMANENTLY EXCLUDED from controller")
        return "\n".join(lines)


def load_config(path: Path) -> dict:
    document=json.loads(path.read_text())
    document.setdefault("kaggle_username",os.getenv("KAGGLE_USERNAME"))
    if not document.get("kaggle_username"): raise ValueError("Configure kaggle_username or KAGGLE_USERNAME")
    return document


def resolve_mode(args) -> str:
    if not any((args.submit, args.status, args.poll, args.download, args.resume)): return "dry-run"
    return "submit" if args.submit else "status" if args.status else "poll" if args.poll else "download" if args.download else "resume"


def main():
    parser=argparse.ArgumentParser(description="Local non-final Phase-3 Kaggle experiment controller")
    modes=parser.add_mutually_exclusive_group(); modes.add_argument("--dry-run",action="store_true"); modes.add_argument("--submit",action="store_true")
    modes.add_argument("--status",action="store_true"); modes.add_argument("--poll",action="store_true"); modes.add_argument("--download",action="store_true"); modes.add_argument("--resume",action="store_true")
    scope=parser.add_mutually_exclusive_group(); scope.add_argument("--stage",choices=STAGES); scope.add_argument("--through",choices=STAGES)
    parser.add_argument("--config",type=Path,default=Path("configs/phase3_kaggle.json")); parser.add_argument("--registry",type=Path,default=Path("artifacts/track5/phase3/registry.json"))
    parser.add_argument("--poll-interval",type=int,default=60); parser.add_argument("--max-polls",type=int,default=60); parser.add_argument("--force",action="store_true"); parser.add_argument("--enable-so400m",action="store_true")
    args=parser.parse_args(); repo=Path(__file__).resolve().parents[1]; config=load_config(repo/args.config if not args.config.is_absolute() else args.config)
    registry=Registry(repo/args.registry if not args.registry.is_absolute() else args.registry); controller=Controller(repo,config,registry,enable_so400m=args.enable_so400m or config.get("enable_so400m",False))
    specs=controller.scoped(args.stage,args.through); stages={spec.stage for spec in specs}; preflight=controller.preflight(stages)
    print("ASSET PREFLIGHT"); [print(f"  {row['asset']}: {'OK' if row['valid'] else 'MISSING/PLACEHOLDER'} {row['reference']}") for row in preflight]
    mode=resolve_mode(args)
    if mode=="dry-run": print(controller.plan(specs)); return
    if mode in {"submit","resume"} and not all(row["valid"] for row in preflight): raise SystemExit("Asset preflight failed; no submission performed")
    def cycle(allow_submit):
        for spec in specs: controller.confirm_existing(spec)
        for spec in specs:
            if registry.data["jobs"][spec.name]["status"] in {"submitted","running"}: controller.refresh_status(spec)
        for spec in specs: controller.download(spec)
        controller.reconcile(remote_mutation=allow_submit)
        if allow_submit:
            for spec in specs:
                status = registry.data["jobs"][spec.name]["status"]
                if controller.ready(spec) or status == "failed": controller.submit(spec,args.force or status == "failed")
    try:
        if mode=="status":
            for spec in specs: controller.refresh_status(spec)
        elif mode=="download":
            for spec in specs: controller.download(spec)
            controller.reconcile(False)
        elif mode=="submit": controller.reconcile(True); [controller.submit(spec,args.force) for spec in specs if controller.ready(spec) or args.force]
        else:
            for index in range(args.max_polls):
                cycle(mode=="resume")
                active=any(registry.data["jobs"][spec.name]["status"] in {"submitted","running"} for spec in specs)
                if not active and not any(controller.ready(spec) for spec in specs): break
                if index+1<args.max_polls: time.sleep(max(args.poll_interval,1))
    except KeyboardInterrupt:
        registry.save(); print("Interrupted safely; rerun with --resume",file=sys.stderr); return
    print(controller.plan(specs))


if __name__=="__main__": main()
