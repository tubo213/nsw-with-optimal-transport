# ruff: noqa
# type: ignore
"""
This module is part of a derivative work based on pyxclib, a library for processing text and performing
various natural language processing tasks.

The original pyxclib can be found at:
https://github.com/pyxclib/pyxclib

Modifications have been made to the original source code to fit the specific needs of this project.

Copyright (C) [Original Copyright Year(s)] by the original authors of pyxclib.
- Original Authors: [List of original authors if known, or "the pyxclib development team"]

This modified work is distributed under the same license as the original, which is the MIT License.

MIT License

Copyright (c) [2024] [Yuki Uehara]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import warnings

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.datasets import load_svmlight_file

__all__ = ["read_data"]


def gen_shape(indices, indptr, zero_based=True):
    _min = min(indices)
    if not zero_based:
        indices = list(map(lambda x: x - _min, indices))
    num_cols = max(indices)
    num_rows = len(indptr) - 1
    return (num_rows, num_cols)


def expand_indptr(num_rows_inferred, num_rows, indptr):
    """Expand indptr if inferred num_rows is less than given"""
    _diff = num_rows - num_rows_inferred
    if _diff > 0:  # Fix indptr as per new shape
        # Data is copied here
        warnings.warn("Header mis-match from inferred shape!")
        return np.concatenate((indptr, np.repeat(indptr[-1], _diff)))
    elif _diff == 0:  # It's fine
        return indptr
    else:
        raise NotImplementedError("Unknown behaviour!")


def ll_to_sparse(X, shape=None, dtype="float32", zero_based=True):
    """Convert a list of list to a csr_matrix; All values are 1.0
    Arguments:
    ---------
    X: list of list of tuples
        nnz indices for each row
    shape: tuple or none, optional, default=None
        Use this shape or infer from data
    dtype: 'str', optional, default='float32'
        datatype for data
    zero_based: boolean or "auto", default=True
        indices are zero based or not

    Returns:
    -------
    X: csr_matrix
    """
    indices = []
    indptr = [0]
    offset = 0
    for item in X:
        if len(item) > 0:
            indices.extend(item)
            offset += len(item)
        indptr.append(offset)
    data = [1.0] * len(indices)
    _shape = gen_shape(indices, indptr, zero_based)
    if shape is not None:
        assert _shape[0] <= shape[0], "num_rows_inferred > num_rows_given"
        assert _shape[1] <= shape[1], "num_cols_inferred > num_cols_given"
        indptr = expand_indptr(_shape[0], shape[0], indptr)
    return csr_matrix(
        (np.array(data, dtype=dtype), np.array(indices), np.array(indptr)), shape=shape
    )


def read_data(filename, header=True, dtype="float32", zero_based=True):
    """Read data in sparse format

    Arguments
    ---------
    filename: str
        output file name
    header: bool, default=True
        If header is present or not
    dtype: str, default='float32'
        data type of values
    zero_based: boolean, default=True
        zwero based indices?

    Returns
    --------
    features: csr_matrix
        features matrix
    labels: csr_matix
        labels matrix
    num_samples: int
        #instances
    num_feat: int
        #features
    num_labels: int
        #labels
    """
    with open(filename, "rb") as f:
        _l_shape = None
        if header:
            line = f.readline().decode("utf-8").rstrip("\n")
            line = line.split(" ")
            num_samples, num_feat, num_labels = int(line[0]), int(line[1]), int(line[2])
            _l_shape = (num_samples, num_labels)
        else:
            num_samples, num_feat, num_labels = None, None, None
        features, labels = load_svmlight_file(
            f, n_features=num_feat, multilabel=True, zero_based=zero_based
        )
        labels = ll_to_sparse(labels, dtype=dtype, zero_based=zero_based, shape=_l_shape)
    return features, labels, num_samples, num_feat, num_labels
