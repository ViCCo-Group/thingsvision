from typing import Any

from .custom import Custom

MODEL_VARIANTS = ["vit_b", "vit_l", "vit_h"]


class SegmentAnything(Custom):
    def __init__(self, device, parameters) -> None:
        super().__init__(device)
        self.backend = "pt"
        self.variant = parameters.get("variant", "vit_h")

    def check_available_variants_and_datasets(self):
        if not self.variant in MODEL_VARIANTS:
            raise ValueError(
                f"\nCould not find Segment Anything model with variant: {self.variant}.\n"
            )

    def create_model(self) -> Any:
        # TODO
        return model, preprocess
