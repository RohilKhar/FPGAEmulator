// picosoc_lite: a self-contained RISC-V SoC built around the picorv32 core.
//
// The raw picorv32 core exposes a 409-bit memory interface that cannot pin out
// on any iCE40 package (see `fpgaforge assess picorv32` -> BLOCKED). A real
// design wraps the core with on-chip memory + peripherals so only a few signals
// leave the chip. Here we expose just clk, resetn, and an 8-bit gpio port.
//
// Internal 512-byte RAM is preloaded with a tiny program that repeatedly writes
// an incrementing counter to a memory-mapped GPIO register, so the gpio output
// counts up as the CPU executes. This lets us bring the CPU up on the virtual
// fabric and verify the flashed bitstream runs the program identically.
//
// picorv32 itself is BSD-licensed by Claire Xen / YosysHQ.
module picosoc_lite (
    input  wire       clk,
    input  wire       resetn,
    output reg  [7:0] gpio
);
    localparam [31:0] GPIO_ADDR = 32'h1000_0000;
    localparam integer RAM_WORDS = 128;   // 512 bytes

    wire        mem_valid;
    wire        mem_instr;
    reg         mem_ready;
    wire [31:0] mem_addr;
    wire [31:0] mem_wdata;
    wire [ 3:0] mem_wstrb;
    reg  [31:0] mem_rdata;

    picorv32 #(
        .ENABLE_COUNTERS   (0),
        .ENABLE_COUNTERS64 (0),
        .COMPRESSED_ISA    (0),
        .CATCH_MISALIGN    (0),
        .CATCH_ILLINSN     (0),
        .ENABLE_IRQ        (0),
        .ENABLE_MUL        (0),
        .ENABLE_DIV        (0),
        .BARREL_SHIFTER    (0),
        .TWO_STAGE_SHIFT   (0),
        .PROGADDR_RESET    (32'h0000_0000),
        .STACKADDR         (32'h0000_0200)
    ) cpu (
        .clk        (clk),
        .resetn     (resetn),
        .trap       (),
        .mem_valid  (mem_valid),
        .mem_instr  (mem_instr),
        .mem_ready  (mem_ready),
        .mem_addr   (mem_addr),
        .mem_wdata  (mem_wdata),
        .mem_wstrb  (mem_wstrb),
        .mem_rdata  (mem_rdata),
        .mem_la_read (),
        .mem_la_write(),
        .mem_la_addr (),
        .mem_la_wdata(),
        .mem_la_wstrb(),
        .pcpi_valid (),
        .pcpi_insn  (),
        .pcpi_rs1   (),
        .pcpi_rs2   (),
        .pcpi_wr    (1'b0),
        .pcpi_rd    (32'b0),
        .pcpi_wait  (1'b0),
        .pcpi_ready (1'b0),
        .irq        (32'b0),
        .eoi        (),
        .trace_valid(),
        .trace_data ()
    );

    reg [31:0] ram [0:RAM_WORDS-1];
    wire [$clog2(RAM_WORDS)-1:0] word_index = mem_addr[$clog2(RAM_WORDS)+1:2];
    wire ram_sel  = mem_valid && (mem_addr < RAM_WORDS*4);
    wire gpio_sel = mem_valid && (mem_addr == GPIO_ADDR);

    integer i;
    initial begin
        for (i = 0; i < RAM_WORDS; i = i + 1) ram[i] = 32'h0000_0000;
        // Program (RV32I), reset PC = 0x0:
        //   00: lui  x2, 0x10000     ; x2 = GPIO base (0x10000000)
        //   04: addi x1, x0, 0       ; x1 = 0
        // loop (0x08):
        //   08: sw   x1, 0(x2)       ; *GPIO = x1
        //   0c: addi x1, x1, 1       ; x1++
        //   10: jal  x0, loop        ; goto 0x08 (offset -8)
        ram[0] = 32'h10000137;
        ram[1] = 32'h00000093;
        ram[2] = 32'h00112023;
        ram[3] = 32'h00108093;
        ram[4] = 32'hff9ff06f;
    end

    always @(posedge clk) begin
        mem_ready <= 1'b0;
        if (!resetn) begin
            gpio <= 8'h00;
        end else if (mem_valid && !mem_ready) begin
            mem_ready <= 1'b1;
            mem_rdata <= ram[word_index];
            if (ram_sel) begin
                if (mem_wstrb[0]) ram[word_index][ 7: 0] <= mem_wdata[ 7: 0];
                if (mem_wstrb[1]) ram[word_index][15: 8] <= mem_wdata[15: 8];
                if (mem_wstrb[2]) ram[word_index][23:16] <= mem_wdata[23:16];
                if (mem_wstrb[3]) ram[word_index][31:24] <= mem_wdata[31:24];
            end
            if (gpio_sel && mem_wstrb[0]) gpio <= mem_wdata[7:0];
        end
    end
endmodule
