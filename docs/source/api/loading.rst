================
Loading & export
================

Load trained models and turn checkpoints into portable artifacts. See
:doc:`../models/pretrained` and :doc:`../models/export` for guides.

Top-level functions
===================

.. currentmodule:: pimm

.. autosummary::
   :toctree: generated
   :nosignatures:

   from_pretrained
   save_pretrained
   push_to_hub

State-dict helpers
==================

Lower-level helpers in :mod:`pimm.export` for partial loads and key
remapping.

.. currentmodule:: pimm.export

.. autosummary::
   :toctree: generated
   :nosignatures:

   load_pretrained
   load_state_dict_from_checkpoint
   load_checkpoint_metadata
   clean_state_dict
   filter_state_dict_by_prefix
   remap_state_dict_keys
