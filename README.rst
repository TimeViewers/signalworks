===========
signalworks
===========

.. image:: https://img.shields.io/pypi/v/signalworks.svg
        :target: https://pypi.python.org/pypi/signalworks

.. image:: https://img.shields.io/travis/lxkain/signalworks.svg
        :target: https://travis-ci.org/lxkain/signalworks

.. image:: https://readthedocs.org/projects/signalworks/badge/?version=latest
        :target: https://signalworks.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status

.. image:: https://pyup.io/repos/github/lxkain/signalworks/shield.svg
     :target: https://pyup.io/repos/github/lxkain/signalworks/
     :alt: Updates


Library to handle signal data and perform signal processing computations


* Free software: MIT license
* Documentation: https://signalworks.readthedocs.io.


Installation
------------

pip install signalworks

Features
--------

* TODO


Testing
-------

Testing dependencies include `git lfs`, `pytest`, `pytest-benchmark`,

To test the library, after cloning the library, run

.. code-block:: bash

    git clone https://github.com/lxkain/signalworks.git
    cd signalworks
    make install
    make test


On windows there is no `Makefile` so you will need to enter commands explicitly


.. code-block:: bash

    git clone https://github.com/lxkain/signalworks.git
    cd signalworks
    git lfs install
    pipenv install --dev --skip-lock .
    git lfs pull
    pipenv run python setup.py test


Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
