// Two clock domains with an UNSYNCHRONIZED crossing: data launched on clk_a is
// captured on clk_b through combinational logic, with no synchronizer. This is
// the classic bug that passes simulation and STA but fails on hardware.
module cdc_unsafe (
    input  wire clk_a,
    input  wire clk_b,
    input  wire rst,
    input  wire d_in,
    output reg  q_out
);
    reg src;            // launched in domain A
    always @(posedge clk_a or posedge rst)
        if (rst) src <= 1'b0;
        else     src <= d_in;

    // Combinational logic on the crossing path, then a single capture flop in B.
    wire crossed = src ^ 1'b1;
    always @(posedge clk_b or posedge rst)
        if (rst) q_out <= 1'b0;
        else     q_out <= crossed;
endmodule
