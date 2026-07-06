// Multiply-accumulate, single combinational stage between registers.
//
// Inputs and output are registered, so the multiply+add forms one long
// register-to-register critical path -> a low Fmax. This is the design the
// optimizer improves (via retiming / abc9 / pipelining / DSP mapping).
//
// Ports are narrow (8/16-bit) so it fits a real iCE40 package for genuine P&R.
module mac (
    input  wire        clk,
    input  wire        rst,
    input  wire [7:0]  a,
    input  wire [7:0]  b,
    input  wire [15:0] c,
    output reg  [15:0] y
);
    reg [7:0]  a_r, b_r;
    reg [15:0] c_r;

    always @(posedge clk) begin
        if (rst) begin
            a_r <= 8'd0;
            b_r <= 8'd0;
            c_r <= 16'd0;
            y   <= 16'd0;
        end else begin
            a_r <= a;
            b_r <= b;
            c_r <= c;
            // Long combinational path: multiply then add, all in one stage.
            y   <= (a_r * b_r) + c_r;
        end
    end
endmodule
