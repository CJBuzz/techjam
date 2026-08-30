import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from scripts.run_phase3_kaggle import (
    Controller,
    FINAL_LOCKED_JOB,
    Registry,
    parse_kaggle_status,
    resolve_mode,
)


class MockRunner:
    def __init__(self): self.commands=[]; self.status="running"; self.on_output=None
    def __call__(self, command, **kwargs):
        self.commands.append(command)
        if command[:3] == ["kaggle","kernels","status"]:
            return subprocess.CompletedProcess(command,0,f'Kernel has status "{self.status}"','')
        if command[:3] == ["kaggle","kernels","output"] and self.on_output:
            self.on_output(Path(command[command.index("-p")+1]))
        return subprocess.CompletedProcess(command,0,"ok","")


def metric_row(candidate_id, field, value, worst=.9):
    return {"candidate_id":candidate_id, field:value, "status":"succeeded",
            "clean_validation_balanced_accuracy":.965,
            "worst_transformed_validation_balanced_accuracy":worst,
            "mean_transformed_validation_balanced_accuracy":worst+.01,
            "inference_multiplier":1,"total_deployment_parameter_count":100,
            "clean_constraint_pass":True}


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.root=Path(self.temp.name)
        self.repo=Path(__file__).resolve().parents[1]; self.runner=MockRunner()
        self.config={"kaggle_username":"tester","source_dataset":"tester/source","training_dataset":"tester/data",
                     "dinov3_asset":"tester/dino","siglip2_large_asset":"tester/siglip",
                     "siglip2_so400m_asset":"tester/so","baseline_clean":.9681,
                     "artifacts_root":str(self.root/"artifacts")}
        self.registry=Registry(self.root/"registry.json")
        self.controller=Controller(self.repo,self.config,self.registry,self.runner)

    def tearDown(self): self.temp.cleanup()

    def test_default_and_dry_run_never_submit(self):
        args=SimpleNamespace(submit=False,status=False,poll=False,download=False,resume=False)
        self.assertEqual(resolve_mode(args),"dry-run")
        plan=self.controller.plan(self.controller.scoped(None,None))
        self.assertIn("FINAL TEST: PERMANENTLY EXCLUDED",plan); self.assertEqual(self.runner.commands,[])

    def test_parallel_r1_and_optional_so(self):
        self.assertTrue(self.controller.ready(self.controller.specs["r1_dinov3"]))
        self.assertTrue(self.controller.ready(self.controller.specs["r1_siglip2_large"]))
        self.assertEqual(self.registry.data["jobs"]["r1_siglip2_so400m"]["status"],"skipped")
        enabled=Controller(self.repo,self.config,Registry(self.root/"so.json"),self.runner,True)
        self.assertTrue(enabled.ready(enabled.specs["r1_siglip2_so400m"]))

    def test_r1_winner_unlocks_r2_and_asset(self):
        self.registry.data["selections"]["r1"]={"status":"selected","job_name":"r1_siglip2_large"}; self.registry.save()
        self.assertTrue(self.controller.ready(self.controller.specs["r2_single"]))
        command=self.controller.generator_command(self.controller.specs["r2_single"])
        self.assertIn("tester/siglip",command); self.assertIn("--r1-kernel-source",command)

    def test_promotion_and_winner_propagation_dependencies(self):
        self.assertFalse(self.controller.ready(self.controller.specs["r2_promotion"]))
        self.registry.data["selections"]["r2"]={"status":"selected","job_name":"r2_compound"}
        self.assertTrue(self.controller.ready(self.controller.specs["r2_promotion"]))
        self.registry.data["jobs"]["r2_promotion"]["status"]="completed"
        self.assertTrue(self.controller.ready(self.controller.specs["r3_mild"]))
        self.registry.data["jobs"]["r3_promotion"]["status"]="completed"
        self.assertTrue(self.controller.ready(self.controller.specs["r4_class_balanced"]))

    def test_r2_promotion_only_after_valid_comparison(self):
        self.registry.data["selections"]["r1"]={"status":"selected","job_name":"r1_dinov3"}
        for name,regime,worst in (("r2_single","single_transform",.90),("r2_compound","compound_curriculum",.92)):
            self.registry.data["jobs"][name]["status"]="completed"
            output=Path(self.registry.data["jobs"][name]["output_download_location"]); output.mkdir(parents=True)
            row=metric_row(name,"regime",regime,worst)
            (output/"r2_summary.json").write_text(json.dumps({"selection_split":"validation","final_test_evaluated":False,
                "eligible_winner":row,"results":[row]}))
        self.controller.reconcile(False)
        self.assertEqual(self.registry.data["selections"]["r2"]["candidate_id"],"r2_compound")
        self.assertTrue(self.controller.ready(self.controller.specs["r2_promotion"]))

    def test_r4_selection_and_independent_r5_r6(self):
        names=("r4_class_balanced","r4_source_balanced","r4_source_quality_matched")
        for index,name in enumerate(names):
            self.registry.data["jobs"][name]["status"]="completed"
            output=Path(self.registry.data["jobs"][name]["output_download_location"]); output.mkdir(parents=True)
            document={"selection_split":"validation","final_test_evaluated":False,
                      "results":[metric_row(name,"bias_policy",name.removeprefix("r4_"),.90+index*.01)]}
            (output/"bias_policy_summary.json").write_text(json.dumps(document))
        self.controller.reconcile(False)
        self.assertEqual(self.registry.data["selections"]["r4"]["status"],"selected")
        self.registry.data["jobs"]["r4_promotion"]["status"]="completed"
        self.assertTrue(self.controller.ready(self.controller.specs["r5_low"]))
        self.assertTrue(self.controller.ready(self.controller.specs["r6_global_only"]))

    def test_r7_uses_actual_upstream_kernel_sources(self):
        for name in ("r3_promotion","r4_promotion","r5_low","r5_high","r5_ensemble","r6_promotion"):
            self.registry.data["jobs"][name]["status"]="completed"
        sources=self.controller.kernel_sources(self.controller.specs["r7_search"])
        self.assertEqual(len(sources),6); self.assertTrue(all(source.startswith("tester/") for source in sources))

    def test_failed_dependency_blocks_but_sibling_survives(self):
        self.registry.data["jobs"]["r5_low"]["status"]="completed"
        self.registry.data["jobs"]["r5_high"]["status"]="failed"
        self.assertEqual(self.registry.data["jobs"]["r5_low"]["status"],"completed")
        self.assertFalse(self.controller.ready(self.controller.specs["r5_ensemble"]))

    def test_completed_job_never_resubmitted(self):
        self.registry.data["jobs"]["r1_dinov3"]["status"]="completed"
        self.assertFalse(self.controller.submit(self.controller.specs["r1_dinov3"]))
        self.assertEqual(self.runner.commands,[])

    def test_status_parsing(self):
        self.assertEqual(parse_kaggle_status('status "complete"'),"complete")
        self.assertEqual(parse_kaggle_status("RUNNING"),"running")
        self.assertEqual(parse_kaggle_status("execution failed"),"failed")

    def test_mocked_status_and_submit_commands(self):
        self.registry.data["jobs"]["r1_dinov3"]["status"]="submitted"; self.runner.status="complete"
        self.assertEqual(self.controller.refresh_status(self.controller.specs["r1_dinov3"]),"complete")
        self.assertEqual(self.registry.data["jobs"]["r1_dinov3"]["remote_status"],"complete")
        self.registry.data["selections"]["r1"]={"status":"selected","job_name":"r1_dinov3"}
        self.assertTrue(self.controller.submit(self.controller.specs["r2_single"]))
        self.assertTrue(any(command[:3]==["kaggle","kernels","push"] for command in self.runner.commands))

    def test_custom_output_download_and_validation(self):
        spec=self.controller.specs["r5_ensemble"]; job=self.registry.data["jobs"][spec.name]
        job.update({"status":"running","remote_status":"complete"})
        def create(output):
            (output/"r5_summary.json").write_text(json.dumps({"selection_split":"validation","final_test_evaluated":False}))
            import numpy as np
            np.savez(output/"val_logits.npz",logits=np.zeros((1,1)),labels=np.zeros(1))
            (output/"recommended_candidate.json").write_text(json.dumps({"selection_split":"validation","final_test_evaluated":False,"candidate":{"id":"x"}}))
        self.runner.on_output=create
        self.assertTrue(self.controller.download(spec)); self.assertEqual(job["status"],"completed")
        self.assertIn(["kaggle","kernels","output",job["kernel_slug"],"-p",job["output_download_location"]],self.runner.commands)

    def test_malformed_completion_is_rejected(self):
        spec=self.controller.specs["r1_dinov3"]; output=Path(self.registry.data["jobs"][spec.name]["output_download_location"])
        output.mkdir(parents=True); (output/"COMPLETED.json").write_text("{}")
        valid,reason,_,_=self.controller.validate_job(spec)
        self.assertFalse(valid); self.assertIn("Incomplete",reason)

    def test_registry_atomic_and_resume_safe(self):
        self.registry.update("r1_dinov3",status="running")
        loaded=Registry(self.registry.path); self.assertEqual(loaded.data["jobs"]["r1_dinov3"]["status"],"running")
        self.assertFalse(self.registry.path.with_suffix(".json.tmp").exists())

    def test_kernel_chaining_and_cli_arguments(self):
        self.registry.data["selections"]["r1"]={"status":"selected","job_name":"r1_dinov3"}
        command=self.controller.generator_command(self.controller.specs["r2_single"])
        index=command.index("--r1-kernel-source")
        self.assertEqual(command[index+1],self.registry.data["jobs"]["r1_dinov3"]["kernel_slug"])

    def test_locked_final_can_never_submit(self):
        with self.assertRaisesRegex(RuntimeError,"ABSOLUTE SAFETY"):
            self.controller.command(["kaggle","kernels","push","-p",f"kaggle/phase3/r7/{FINAL_LOCKED_JOB}"])
        self.assertNotIn(FINAL_LOCKED_JOB,self.controller.specs)
        with self.assertRaisesRegex(RuntimeError,"ABSOLUTE SAFETY"):
            self.controller.command(["kaggle","kernels","push","-p","kaggle/phase3/r7/r7-locked-final"])


if __name__=="__main__": unittest.main()
