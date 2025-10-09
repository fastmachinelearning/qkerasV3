# Copyright 2020 Google LLC
#
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Creates a random forest to generate hardware for it."""

import pickle

import keras.ops.numpy as knp

from .random_tree import RandomTree


def fit_parallel(max_depth, min_size, sample, mask_stuck_at_values):
    tree = RandomTree(max_depth, min_size)
    tree.fit(sample, mask_stuck_at_values)

    return tree


class RandomForest:
    def __init__(self, max_depth, min_size, n_trees, use_mean=False, sample_size=None):
        self.max_depth = max_depth
        self.min_size = min_size
        self.use_mean = use_mean
        self.sample_size = sample_size
        self.n_trees = n_trees
        self.inputs = None
        self.bits = None
        self.is_neg = None

        self.trees = None

    @staticmethod
    def save(model, filename):
        """Saves model to disk."""
        print(f"... saving model in {filename}")
        f = open(filename, "wb")
        pickle.dump(model, f)
        f.close()

    @staticmethod
    def load(filename):
        """Loads model from disk."""
        print(f"... loading model from {filename}")
        f = open(filename, "rb")
        random_forest = pickle.load(f)
        f.close()

        return random_forest

    def subsample(self, dataset):
        """Subsamples dataset if we do not want to use entire dataset."""
        sample_idx = np.random.choice(dataset.shape[0], self.sample_size, replace=True)
        sample = dataset[sample_idx, ...]
        return sample

    def fit(self, dataset, verbose=False):
        """Fits random tree to model."""
        self.inputs = dataset.shape[1] - 1
        self.bits = knp.ceil(
            knp.log2(knp.abs(knp.amax(dataset, axis=0) - knp.amin(dataset, axis=0)))
        ).astype(np.int32)
        self.is_neg = (knp.amin(dataset, axis=0) < 0).astype(np.int8)

        self.trees = []

        for i in range(self.n_trees):
            if verbose:
                print(f"... creating tree {i}")

            # as subsample is an expensive operation, we will only perform it if it
            # reduces the dataset substantially

            if self.sample_size and self.sample_size < 0.3 * dataset.shape[0]:
                if verbose:
                    print(f"... generated subsample of size {self.sample_size}")
                sample = self.subsample(dataset)
            else:
                sample = dataset

            self.trees.append(fit_parallel(self.max_depth, self.min_size, sample, True))

    def predict_row(self, row):
        """Predicts output for single row."""
        result = [tree.predict_row(row) for tree in self.trees]
        if self.use_mean:
            return int(knp.round(knp.mean(result)))
        else:
            return max(set(result), key=result.count)

    def predict(self, data):
        """Predicts class based on data."""

        assert self.trees is not None

        return knp.array([self.predict_row(data[i]) for i in range(data.shape[0])])

    def gen_code(self, filename, func_name):
        """Generates code for model."""

        assert self.bits is not None

        vd_list = []
        n_vars = 0
        for tree in self.trees:
            vd_list.append(tree.gen_code(n_vars))
            n_vars += len(vd_list[-1])

        # checks the type by the suffix

        is_v = filename.split(".")[-1] == "v"

        assert self.inputs

        f = open(filename, "w")

        i_bits = knp.sum(self.bits[:-1])
        o_bits = self.bits[-1]
        o_sign = self.is_neg[-1]

        if is_v:
            f.write(
                f"module {func_name}(input [{i_bits - 1}:0] i, output [{o_bits - 1}:0] o);\n"
            )
        else:
            f.write("#include<ac_int.h>\n\n")
            f.write(
                f"void {func_name}(ac_int<{i_bits},false> i, ac_int<{o_bits},{o_sign}> &o)\n"
            )
            f.write("{\n")

        # write function headline
        s_in_line = []

        i_bits = self.bits[0]
        i_sign = self.is_neg[0]

        if is_v:
            i_datatype = "  wire {}[{}:0] ".format(
                "signed " if i_sign else "", i_bits - 1
            )
        else:
            i_datatype = f"  ac_int<{i_bits},{i_sign}> "

        len_s = len(i_datatype)

        for i in range(self.inputs):
            if is_v:
                s = (
                    "i_"
                    + str(i)
                    + " = "
                    + "i["
                    + str(i_bits * (i + 1) - 1)
                    + ":"
                    + str(i_bits * i)
                    + "]"
                )
            else:
                s = (
                    "i_"
                    + str(i)
                    + " = "
                    + "i.slc<"
                    + str(i_bits)
                    + ">("
                    + str(i_bits * i)
                    + ")"
                )
            if (
                len_s + len(s) + 2 > 70
                or i_bits != self.bits[i]
                or i_sign != self.is_neg[i]
            ):
                f.write(i_datatype + ", ".join(s_in_line) + ";\n")

                s_in_line = []
                if is_v:
                    i_datatype = "  wire {}[{}:0] ".format(
                        "signed " if i_sign else "", i_bits - 1
                    )
                else:
                    i_datatype = f"  ac_int<{i_bits},{i_sign}> "

                len_s = len(i_datatype)

            s_in_line.append(s)
            len_s += len(s) + 2

        if s_in_line:
            f.write(i_datatype + ", ".join(s_in_line) + ";\n")

        if is_v:
            o_datatype = "  wire {}[{}:0] ".format("signed " if o_sign else "", o_bits)
        else:
            o_datatype = f"  ac_int<{o_bits},{o_sign}> "

        o_list = []
        for i in range(len(vd_list)):
            for v in vd_list[i]:
                if is_v:
                    f.write(o_datatype + v + " = " + vd_list[i][v] + ";\n")
                else:
                    f.write(o_datatype + v + " = " + vd_list[i][v] + ";\n")
            f.write("\n")
            o_list.append(v)

        assert len(o_list) <= 3

        if is_v:
            f.write("  assign ")
        else:
            f.write("  ")

        if len(o_list) == 1:
            f.write("o = " + o_list[0] + ";")
        elif len(o_list) == 2:
            cond = "( " + o_list[0] + " == " + o_list[1] + " ) "
            n1 = o_list[0]
            n0 = "( ( " + " + ".join(o_list) + " ) >> 1 )"
            f.write("o = " + cond + "? " + n1 + ": " + n0)
        elif len(o_list) == 3:
            cond = (
                "( "
                + "( "
                + " == ".join(o_list[0:2])
                + " )?"
                + o_list[0]
                + ":"
                + "( "
                + " == ".join(o_list[1:])
                + " )?"
                + o_list[1]
                + ":"
                + "( "
                + " == ".join([o_list[0], o_list[2]])
                + " )?"
                + o_list[0]
                + ":"
                + "( "
                + " < ".join(o_list[0:2])
                + " ) ?"
                + "( ( "
                + " < ".join(o_list[1:])
                + " ) ?"
                + o_list[1]
                + ":"
                + o_list[2]
                + " ) : "
                + "( ( "
                + " < ".join([o_list[0], o_list[2]])
                + " ) ?"
                + o_list[0]
                + ":"
                + o_list[2]
                + " )"
            )
            f.write("o = " + cond + ";\n")
        if is_v:
            f.write("endmodule")
        else:
            f.write("}")

        f.close()
