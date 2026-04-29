"""qrackbind PennyLane integration — device plugins.

Devices
-------
- ``QrackDevice``                  → ``qrackbind.simulator``
- ``QrackStabilizerDevice``        → ``qrackbind.stabilizer``
- ``QrackStabilizerHybridDevice``  → ``qrackbind.stabilizer_hybrid``
"""

from qrackbind.pennylane.device import QrackDevice
from qrackbind.pennylane.stabilizer_device import (
    QrackStabilizerDevice,
    QrackStabilizerHybridDevice,
)

__all__ = [
    "QrackDevice",
    "QrackStabilizerDevice",
    "QrackStabilizerHybridDevice",
]
