# Copyright (c) Facebook Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""isort:skip_file"""

from .transformer_config import (
    TransformerConfig,
    DEFAULT_MAX_SOURCE_POSITIONS,
    DEFAULT_MAX_TARGET_POSITIONS,
    DEFAULT_MIN_PARAMS_TO_WRAP,
)
from .transformer_decoder import TransformerDecoder, TransformerDecoderBase, Linear
from .transformer_encoder import TransformerEncoder, TransformerEncoderBase
from .transformer_legacy import (
    TransformerModel,
    base_architecture,
    tiny_architecture,
    transformer_iwslt_de_en_1,
    transformer_iwslt_de_en_2,
    transformer_iwslt_de_en_3,
    transformer_iwslt_de_en_4,
    transformer_iwslt_de_en_5,
    transformer_iwslt_de_en_6,
    transformer_wmt_en_de,
    transformer_vaswani_wmt_en_de_big,
    transformer_vaswani_wmt_en_fr_big,
    transformer_wmt_en_de_big,
    transformer_wmt_en_de_big_t2t,
)
from .transformer_base import TransformerModelBase, Embedding


__all__ = [
    "TransformerModelBase",
    "TransformerConfig",
    "TransformerDecoder",
    "TransformerDecoderBase",
    "TransformerEncoder",
    "TransformerEncoderBase",
    "TransformerModel",
    "Embedding",
    "Linear",
    "base_architecture",
    "tiny_architecture",
    "transformer_iwslt_de_en_1",
    "transformer_iwslt_de_en_2",
    "transformer_iwslt_de_en_3",
    "transformer_iwslt_de_en_4",
    "transformer_iwslt_de_en_5",
    "transformer_iwslt_de_en_6",
    "transformer_wmt_en_de",
    "transformer_vaswani_wmt_en_de_big",
    "transformer_vaswani_wmt_en_fr_big",
    "transformer_wmt_en_de_big",
    "transformer_wmt_en_de_big_t2t",
    "DEFAULT_MAX_SOURCE_POSITIONS",
    "DEFAULT_MAX_TARGET_POSITIONS",
    "DEFAULT_MIN_PARAMS_TO_WRAP",
]
