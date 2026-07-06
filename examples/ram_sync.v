// Small synchronous single-port RAM (register-based, infers BRAM).
//
// Used to demonstrate formal equivalence on a memory-bearing design: the
// SAT (bit-blasting) engine explodes as the memory grows, while the SMT
// engine models the memory as an array and scales.
module ram_sync #(
    parameter AW = 4,
    parameter DW = 8
) (
    input  wire           clk,
    input  wire           we,
    input  wire [AW-1:0]  addr,
    input  wire [DW-1:0]  din,
    output reg  [DW-1:0]  dout
);
    reg [DW-1:0] mem [0:(1<<AW)-1];

    always @(posedge clk) begin
        if (we)
            mem[addr] <= din;
        dout <= mem[addr];
    end
endmodule
