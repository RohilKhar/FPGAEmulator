from fpgaforge import features as feat


def test_feature_vector_schema_length():
    vec = feat.to_vector(feat.empty_features())
    assert len(vec) == len(feat.FEATURE_NAMES)


def test_from_yosys_json_counts_cells():
    netlist = {
        "modules": {
            "mac": {
                "ports": {
                    "a": {"direction": "input", "bits": [2, 3]},
                    "y": {"direction": "output", "bits": [4, 5]},
                },
                "cells": {
                    "l0": {"type": "SB_LUT4", "connections": {"O": [4], "I0": [2]}},
                    "d0": {"type": "SB_DFF", "connections": {"Q": [5], "D": [4]}},
                    "c0": {"type": "SB_CARRY", "connections": {"CO": [6], "CI": [2]}},
                    "m0": {"type": "SB_MAC16", "connections": {"O": [7], "A": [3]}},
                },
                "memories": {},
            }
        }
    }
    f = feat.from_yosys_json(netlist, "mac")
    assert f["num_cells"] == 4
    assert f["num_luts"] == 1
    assert f["num_ffs"] == 1
    assert f["num_carries"] == 1
    assert f["num_dsp"] == 1
    assert f["num_inputs"] == 1
    assert f["num_outputs"] == 1


def test_from_rtl_text_detects_arithmetic():
    rtl = """
    module mac(input clk, input [15:0] a, input [15:0] b,
               input [31:0] c, output reg [31:0] y);
      always @(posedge clk) y <= a * b + c;
    endmodule
    """
    f = feat.from_rtl_text(rtl)
    assert f["num_dsp"] >= 1     # a * b
    assert f["num_inputs"] > 0
    assert f["num_outputs"] > 0
