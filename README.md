# BREP translated to Python+NEURON

## Author info

- Author: Ohki Katakura
- Email: contact@neuronalpail.com
- Facility: Centre for Computer Science and Informatics Research (CCSIR), University of Hertfordshire
- Date of Creation: 31 March 2021

## Introduction

This is a translation of the Boundary Representation Language (BREP) [1] from [Chicken Scheme](https://www.call-cc.org) to Python+NEURON.

The original script was written by Ivan Raikov and available as a part of [Model of the cerebellar granular network](https://senselab.med.yale.edu/ModelDB/showModel.cshtml?model=232023) (the GL model in following) [2].
The original script cannot compile with latest Chicken Scheme (version 5).

In this project, I aim to create proper translation of BREP to Python+NEURON, which is easier to read for more people and compatible with less platforms.
The BREP works as a part of the GL model which require NEURON simulator, Python and MPI.
This translation works with the same environment, but requires additional package of Python, `scipy`.

## Original BREP

<!-- I copied BREP and made several modification to original BREP to output
- progress messages
- random numbers
for debugging reasons.

Moreover, with MPI version 3, `mpi` package of Chicken Scheme 4 cannot compiled.
For more safety compilation of BREP, I modified script to deploy/compile it.

With my modification, I confirmed the script work in my local computer (AMD Ryzen 5 3600; x86_64 architecture, 6 cores). -->

The original BREP is protected with copyright.
The script can be obtained from [the GL model](https://senselab.med.yale.edu/ModelDB/showModel.cshtml?model=232023).

## Requirements

Translated code requires

- **[Python](https://www.python.org)** (tested with version 3.8.6)
- **[NEURON simulator](https://neuron.yale.edu/)** (tested with version 7.8.2 HEAD)
- **[MPI](https://www.mpi-forum.org)** (tested with [Open MPI](https://www.open-mpi.org) 4.1.0)

and Python packages

- `mpi4py` (tested with version 3.0.3)
- `NEURON` (tested with version 7.8.2)
- `numpy` (tested with version 1.20.1)
- `scipy` (tested with version 1.6.1)

the author managed the packages with `pip` (version 21.0.1).

<!-- For original BREP, additionally **[Chicken Scheme](https://www.call-cc.org)** version 4 (tested with version 4.8.0) is required.
This is an outdated version but several packages of BREP are not compatible Chicken Scheme 5. -->

Again, BREP works as a part of [the GL model](https://senselab.med.yale.edu/ModelDB/showModel.cshtml?model=232023).
For the reason, the model is also required for full test.

## Usage

Under construction.

## References

1. Raikov, I., Kumar, S.S., Torben-Nielsen, B., De Schutter, E., 2014. A NineML-based domain-specific language for computational exploration of connectivity in the cerebellar granular layer. BMC Neuroscience 15, P176. https://doi.org/10.1186/1471-2202-15-S1-P176
2. Sudhakar, S.K., Hong, S., Raikov, I., Publio, R., Lang, C., Close, T., Guo, D., Negrello, M., De Schutter, E., 2017. Spatiotemporal network coding of physiological mossy fiber inputs by the cerebellar granular layer. PLOS Computational Biology 13, e1005754. https://doi.org/10.1371/journal.pcbi.1005754

## Copyright

Copyright 2021 Ohki Katakura

BREP-Python is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This programme is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this programme. If not, see <https://www.gnu.org/licenses/>.

## Update log of the README

- 31 March 2021: First publication.