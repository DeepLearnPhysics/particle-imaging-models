====================
Core building blocks
====================

The shared abstractions used across pimm: config-driven builders,
:class:`~pimm.models.utils.structure.Point`,
:class:`~pimm.utils.config.Config`, :class:`~pimm.utils.registry.Registry`, and
checkpoint/resume primitives.

Builders
========

Construct registered objects from config dictionaries. See the per-registry
pages (:doc:`Models <registry/models>`, :doc:`Datasets <registry/datasets>`, …)
for what each registry contains.

.. currentmodule:: pimm.models.builder

.. autosummary::
   :toctree: generated
   :nosignatures:

   build_model

.. currentmodule:: pimm.datasets.builder

.. autosummary::
   :toctree: generated
   :nosignatures:

   build_dataset

.. currentmodule:: pimm.models.losses.builder

.. autosummary::
   :toctree: generated
   :nosignatures:

   build_criteria

.. autosummary::
   :toctree: generated
   :template: pimm_class.rst
   :nosignatures:

   Criteria

Builder registries
------------------

.. currentmodule:: pimm.models.builder

.. py:data:: MODELS
   :type: pimm.utils.registry.Registry

   Registry used by :func:`build_model` to resolve ``model.type`` strings.

.. currentmodule:: pimm.models.losses.builder

.. py:data:: LOSSES
   :type: pimm.utils.registry.Registry

   Registry used by :func:`build_criteria` to resolve each
   ``criteria[].type`` string.

Point structures & base modules
===============================

.. currentmodule:: pimm.models.utils.structure

.. autosummary::
   :toctree: generated
   :template: pimm_class.rst
   :nosignatures:

   Point

.. currentmodule:: pimm.models.modules

.. autosummary::
   :toctree: generated
   :template: pimm_class.rst
   :nosignatures:

   PointModule
   PointSequential
   PointModel

Configuration & registry
========================

.. currentmodule:: pimm.utils.config

.. autosummary::
   :toctree: generated
   :template: pimm_class.rst
   :nosignatures:

   Config

.. currentmodule:: pimm.utils.registry

.. autosummary::
   :toctree: generated
   :template: pimm_class.rst
   :nosignatures:

   Registry

Checkpointing & resume
======================

The checkpoint manager and the resume-state schema. See
:doc:`../operations/checkpoints`.

.. currentmodule:: pimm.utils.checkpoints

.. autosummary::
   :toctree: generated
   :template: pimm_class.rst
   :nosignatures:

   CheckpointManager

.. currentmodule:: pimm.engines._train_utils

.. autosummary::
   :toctree: generated
   :template: pimm_class.rst
   :nosignatures:

   TrainState

Data loading & collation
=========================

The transform composition and collation helpers that turn samples into packed
point-cloud batches, the packed/padded conversion used by PoLAr-MAE, and the
checkpointable sampler that lets the training loader resume mid-epoch. See
:doc:`../extend/add_dataset` and :doc:`../data/custom`.

.. currentmodule:: pimm.datasets.transform.base

.. autosummary::
   :toctree: generated
   :template: pimm_class.rst
   :nosignatures:

   Compose

.. currentmodule:: pimm.datasets.utils

.. autosummary::
   :toctree: generated
   :nosignatures:

   collate_fn

.. currentmodule:: pimm.models.polarmae.data

.. autosummary::
   :toctree: generated
   :nosignatures:

   packed_to_batched

.. currentmodule:: pimm.datasets.stateful

.. autosummary::
   :toctree: generated
   :template: pimm_class.rst
   :nosignatures:

   StatefulRandomSampler

Engine setup
============

Derive per-rank loader settings, initialize reproducibility controls, and
materialize the runtime config used by the trainer.

.. currentmodule:: pimm.engines.defaults

.. autosummary::
   :toctree: generated
   :nosignatures:

   default_setup

Hook lifecycle
==============

The base lifecycle interface shared by every registered training hook. Concrete
hook implementations are listed in :doc:`registry/hooks`.

.. currentmodule:: pimm.engines.hooks.default

.. autosummary::
   :toctree: generated
   :template: pimm_class.rst
   :nosignatures:

   HookBase

Distributed helpers
===================

Process-group setup and small collectives (:mod:`pimm.utils.comm`).

.. currentmodule:: pimm.utils.comm

.. autosummary::
   :toctree: generated
   :nosignatures:

   get_world_size
   get_rank
   is_main_process
   synchronize
   all_gather
   reduce_dict
