"""Schema versions for the frozen contracts.

A hashed schema is never mutated — it gains a version (D32). The canonical hash
covers every serialized field, so adding a field to an existing version would
change the digest of proposals that were already hashed: stored hashes would stop
reproducing, and any token minted over one would become unverifiable. So each
version is its own class, and old versions stay parseable forever.

``schema_version`` is *inside* the hash, which is what makes a cross-version
collision impossible by construction: the same business fields under 1.0 and 1.1
are different documents with different digests, so an approval of one can never
be replayed as an approval of the other.
"""

from typing import Literal

SCHEMA_VERSION_1_0: Literal["1.0"] = "1.0"
SCHEMA_VERSION_1_1: Literal["1.1"] = "1.1"

#: The version this codebase *produces*. Older versions remain readable.
SCHEMA_VERSION: Literal["1.1"] = SCHEMA_VERSION_1_1
