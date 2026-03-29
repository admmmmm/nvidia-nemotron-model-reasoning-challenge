from __future__ import annotations

from src.data.synthetic_public_like.bit_manipulation import make_bit_manipulation_puzzle
from src.data.synthetic_public_like.cipher import make_cipher_puzzle
from src.data.synthetic_public_like.equation_numeric import make_equation_numeric_puzzle
from src.data.synthetic_public_like.equation_symbolic import make_equation_symbolic_puzzle
from src.data.synthetic_public_like.unit_conversion import make_unit_conversion_puzzle


GENERATOR_BUILDERS = {
    "unit_conversion": make_unit_conversion_puzzle,
    "bit_manipulation": make_bit_manipulation_puzzle,
    "cipher": make_cipher_puzzle,
    "equation_numeric": make_equation_numeric_puzzle,
    "equation_symbolic": make_equation_symbolic_puzzle,
}
