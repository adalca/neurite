# ---- neuron ----
import sys
print('the `neuron` library has been renamed to `neurite` to avoid pypi conflicts.\n' \
      'neuron imports will be depricated. Please switch to importing neurite.', \
          file=sys.stderr)

from neurite import *