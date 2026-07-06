// Pipelined multiply-accumulate: the multiply and the add live in separate
// register-to-register stages, shortening the critical path and raising Fmax
// at the cost of one extra cycle of latency.
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
    reg [15:0] prod;
    reg [15:0] c_d;

    always @(posedge clk) begin
        if (rst) begin
            a_r  <= 8'd0;
            b_r  <= 8'd0;
            c_r  <= 16'd0;
            prod <= 16'd0;
            c_d  <= 16'd0;
            y    <= 16'd0;
        end else begin
            a_r  <= a;
            b_r  <= b;
            c_r  <= c;
            prod <= a_r * b_r;  // stage 1: multiply
            c_d  <= c_r;
            y    <= prod + c_d; // stage 2: add
        end
    end
endmodule
