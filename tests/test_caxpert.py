#!/usr/bin/env python

"""Tests for `caxpert` package."""

import pytest
from ase.build import fcc111, molecule, add_adsorbate
from caxpert.src.tasks.gen_str import generate_structures

def test_generate_structures_one_ads():
    """
    Test the generation of structures using the ICET tool.
    """
    # Create a primitive structure
    prim_structure = fcc111('Ni',size=(1,1,4), vacuum=13)
    prim_structure.set_tags([0 for i in range(len(prim_structure))])
    # Create an adsorbate structure
    adsorbate = molecule('CO', vacuum=13, tags=[2,2])
    # Create a tuple of the adsorbate and the index of the binding atom
    adsorbate_list = [(adsorbate, 1)]
    add_adsorbate(prim_structure, adsorbate, 1.8, offset=(0, 0), mol_index=1)
    # Create a list of the indices of the center atoms of the adsorbates
    ads_center_atom_ids = [a.index for a in prim_structure if a.symbol == 'C']
    # Set the cell size
    cell_size = 4
    # Generate the structures
    generate_structures(prim_structure, adsorbate_list, ads_center_atom_ids, cell_size)
    # # Check if the database is created
    # assert os.path.exists('init_structures.db')

def test_generate_structures_two_ads():
    """
    Test the generation of structures using the ICET tool.
    """
    # Create a primitive structure
    prim_structure = fcc111('Ni',size=(1,1,4), vacuum=13)
    prim_structure.set_tags([0 for i in range(len(prim_structure))])
    # Create an adsorbate structure
    co = molecule('CO', vacuum=13, tags=[2,2])
    h = molecule('H', vacuum=13, tags=[2])
    # Create a tuple of the adsorbate and the index of the binding atom
    adsorbate_list = [(co, 1), (h, 0)]
    add_adsorbate(prim_structure, co, 1.8, offset=(0, 0), mol_index=1)
    # Create a list of the indices of the center atoms of the adsorbates
    ads_center_atom_ids = [a.index for a in prim_structure if a.symbol == 'C']
    # Set the cell size
    cell_size = 4
    # Generate the structures
    generate_structures(prim_structure, adsorbate_list, ads_center_atom_ids, cell_size)
    # # Check if the database is created
    # assert os.path.exists('init_structures.db')

# test_generate_structures_one_ads()
test_generate_structures_two_ads()
# @pytest.fixture
# def response():
#     """Sample pytest fixture.

#     See more at: http://doc.pytest.org/en/latest/fixture.html
#     """
#     # import requests
#     # return requests.get('https://github.com/audreyr/cookiecutter-pypackage')


# def test_content(response):
#     """Sample pytest test function with the pytest fixture as an argument."""
#     # from bs4 import BeautifulSoup
#     # assert 'GitHub' in BeautifulSoup(response.content).title.string
