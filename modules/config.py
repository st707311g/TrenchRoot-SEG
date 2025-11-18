from pathlib import Path
from typing import Final

app_name: Final = "TrenchRoot-SEG"
version: Final = "v1.5.0"

training_data_dir: Final = Path("data") / "for_train"
root_image_dir: Final = training_data_dir / "root_images"
separated_root_image_dir: Final = training_data_dir / "root_images_separated"
trace_image_dir: Final = training_data_dir / "trace_images"
separated_trace_image_dir: Final = training_data_dir / "trace_images_separated"
