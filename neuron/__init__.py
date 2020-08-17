# ---- neuron ----
import sys
import inspect

print('The `neuron` library has been renamed to `neurite` to avoid pypi conflicts.\n' \
      'neuron imports will be depricated. Please switch to importing neurite.', \
      file=sys.stderr)

# print file that imports neuron
frame = [s for s in inspect.stack(0) if s.function == '<module>'][1]
print('INFO: neuron was imported from %s on line %d.' % (frame.filename, frame.lineno),
      file=sys.stderr)

from neurite import *
