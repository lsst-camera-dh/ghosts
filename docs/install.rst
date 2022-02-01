.. _install:

Installing ghosts
====================

At some point you will be able to install `ghosts` with `pip` or from source.
`ghosts` depends upon `batoid`, call it even just an extension if you wish.


Installing with pip
-------------------

This is will work one day, believe me.

.. code-block:: bash

    pip install ghosts



Installing from source
-----------------------

This is the recommended installation procedure in a conda environment.
It shall take care of the `batoid` dependency cleanly.

.. code-block:: bash

    git clone https://github.com/bregeon/ghosts.git
    cd ghosts
    conda env create -f environment.yml
    conda activate ghosts
    pip install -r requirements.txt
    pip install -e .

