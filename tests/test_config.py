from pathlib import Path

from critter_capture.config import load_config


def test_config_environment_merge(tmp_path: Path) -> None:
    config_text = """
base:
  environment: local
  data:
    csv_path: /tmp/example.csv
    uuid_column: uuid
    image_url_column: image_url
    label_column: taxon_id
    label_names_column: common_name
  model:
    num_classes: null
    conv_filters: [16, 32, 64, 64, 64]
  training:
    epochs: 1
  tuning:
    enabled: false
  deployment:
    enable: false
  inference:
    batch_size: 1
  logging:
    level: INFO
  storage:
    mlflow_tracking_uri: http://localhost:5000
environments:
  test:
    training:
      epochs: 2
"""
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(config_text)

    cfg = load_config(cfg_path, "test")
    assert cfg.training.epochs == 2
    assert cfg.data.csv_path == Path("/tmp/example.csv")
