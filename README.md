RatPred 🐀
=========

_Rat(ing) Pred(ictor) – A Referenceless NLG Quality Estimation Tool_

RatPred predicts NLG quality ratings/ranks using a recurrent neural network. 
It is trained using a set of human-rated/ranked NLG outputs along with the corresponing 
source meaning representations (MRs). It can then estimate human ratings/ranks given
a MR and an NLG system output only. Unlike most automated metrics used for NLG, 
such as BLEU or NIST, RatPred does not need human-authored reference texts.

For details on the system architecture and its performance, please refer to
our [ICML-LGNL paper](https://arxiv.org/abs/1708.01759).

Note that RatPred is highly experimental and only tested, so bugs are inevitable. If you find a bug, feel free to [contact me](https://github.com/tuetschek) or [open an issue](https://github.com/UFAL-DSG/ratpred/issues). 

Installation and Usage
----------------------

See [USAGE.md](USAGE.md). Note that currently RatPred is using Python 2. Python 3 version is in the works.

Citing
------

If you use or refer to RatPred, please cite [our ICML-LGNL paper](https://arxiv.org/abs/1708.01759):

Ondřej Dušek, Jekaterina Novikova, and Verena Rieser (2017): Referenceless Quality Estimation for Natural Language Generation. In _Proceedings of the 1st Workshop on Learning to Generate Natural Language_, Sydney, Australia.

License
-------

Author: [Ondřej Dušek](https://github.com/tuetschek)

Copyright © 2017-2018 Interaction Lab, Heriot-Watt University, Edinburgh.  
Copyright © 2019 Institute of Formal and Applied Linguistics, Charles University, Prague.


Licensed under the Apache License, Version 2.0 (see [LICENSE.txt](LICENSE.txt)).

Acknowledgements
----------------

This research received funding from the EPSRC projects  DILiGENt (EP/M005429/1) and  MaDrIgAL (EP/N017536/1) and Charles University project PRIMUS/19/SCI/10. The Titan Xp used for this research was donated by the NVIDIA Corporation.


