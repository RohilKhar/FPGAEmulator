// A tiny "real board" design for the virtual-board emulator:
//   * blinks an LED,
//   * continuously transmits "Hi\n" over a UART TX pin.
//
// The UART bit rate is clk / DIV. Run the virtual board at clock_mhz with
// baud = clock_mhz*1e6 / DIV so the board's decoder samples in step. With the
// defaults (DIV=12, clock_mhz=12) that is baud = 1_000_000.
module blinky_uart (
    input  wire clk,
    input  wire resetn,     // active-low reset
    output reg  led,
    output reg  tx
);
    localparam integer DIV = 12;        // clocks per UART bit

    // 3-byte message "Hi\n" exposed as a tiny combinational ROM (no BRAM).
    function [7:0] rom;
        input [1:0] i;
        case (i)
            2'd0: rom = 8'h48;          // 'H'
            2'd1: rom = 8'h69;          // 'i'
            default: rom = 8'h0A;       // '\n'
        endcase
    endfunction

    reg [3:0] divcnt;
    reg [3:0] bitpos;                   // 0=start, 1..8=data, 9=stop
    reg [1:0] idx;
    reg [7:0] shifter;
    reg [15:0] blink;

    always @(posedge clk) begin
        if (!resetn) begin
            tx      <= 1'b1;            // UART idle-high
            led     <= 1'b0;
            divcnt  <= 4'd0;
            bitpos  <= 4'd0;
            idx     <= 2'd0;
            shifter <= 8'd0;
            blink   <= 16'd0;
        end else begin
            // LED blink.
            blink <= blink + 1'b1;
            if (blink == 16'd63) begin
                led   <= ~led;
                blink <= 16'd0;
            end

            // UART transmit, one bit every DIV clocks.
            if (divcnt == DIV - 1) begin
                divcnt <= 4'd0;
                case (bitpos)
                    4'd0: begin                 // start bit
                        tx      <= 1'b0;
                        shifter <= rom(idx);
                        bitpos  <= 4'd1;
                    end
                    4'd1, 4'd2, 4'd3, 4'd4,
                    4'd5, 4'd6, 4'd7, 4'd8: begin   // 8 data bits, LSB first
                        tx      <= shifter[0];
                        shifter <= shifter >> 1;
                        bitpos  <= bitpos + 1'b1;
                    end
                    default: begin              // stop bit -> next byte
                        tx     <= 1'b1;
                        bitpos <= 4'd0;
                        idx    <= (idx == 2'd2) ? 2'd0 : idx + 1'b1;
                    end
                endcase
            end else begin
                divcnt <= divcnt + 1'b1;
            end
        end
    end
endmodule
