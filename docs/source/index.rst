qkerasV3 Documentation
=====================

qkerasV3 is a Keras 3–compatible continuation of QKeras, focused on quantization-aware training and model compression.

.. important::

   qkerasV3 currently supports **TensorFlow** as the Keras backend.
   Set the backend before importing:

   .. code-block:: bash

      export KERAS_BACKEND=tensorflow

   In the current version v1.1.x AutoQKeras is not working.
   There will be an update in the the future to support this feature again using the sklearn gridsearch API.

.. toctree::
   :maxdepth: 2
   :caption: Getting started

   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: Guides

   examples/index
   notebooks

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api/index
