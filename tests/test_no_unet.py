from pathlib import Path
def test_runner_does_not_load_unet_or_pipeline():
    text=(Path(__file__).parents[1]/"scripts/run_severity_matrix.py").read_text()
    assert "StableDiffusionPipeline" not in text
    assert "UNet2DConditionModel" not in text
    assert ".unet" not in text
    assert "AutoencoderKL" in text
