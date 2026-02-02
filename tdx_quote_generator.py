import ctypes
import sys
import base64
import os
import random

tdx_attest = ctypes.CDLL("/usr/lib/x86_64-linux-gnu/libtdx_attest.so")

# Constants from tdx_attest.h
TDX_UUID_SIZE = 16

# Not sure if both of these should be change based on the size of the report
# Using the value from tdx_attest.h for now
TDX_REPORT_DATA_SIZE = 64
TDX_REPORT_SIZE = 1024 

# Structs from tdx_attest.h
class TdxUuid(ctypes.Structure):
    _fields_ = [("d", ctypes.c_uint8 * TDX_UUID_SIZE)]

class TdxReportData(ctypes.Structure):
    _fields_ = [("d", ctypes.c_uint8 * TDX_REPORT_DATA_SIZE)]

class TdxReport(ctypes.Structure):
    _fields_ = [("d", ctypes.c_uint8 * TDX_REPORT_SIZE)]


# This is from test_tdx_attest.c to populate the TDX report. Only for testing
def gen_report_data():
    return (ctypes.c_uint8 * TDX_REPORT_DATA_SIZE)(*random.randbytes(TDX_REPORT_DATA_SIZE))

# This is from test_tdx_attest.c, convert buffer to hex dump
def print_hex_dump(title, buf):
    print(f"\n\t\t{title}")
    for i in range(0, len(buf), 16):
        hex_row = ' '.join(f"{b:02x}" for b in buf[i:i+16])
        print(f"{i:08x}: {hex_row}")

def generate_quote(payload_hash, output_dir="./"):

    # Data type needed for quote generation
    selected_att_key_id = TdxUuid()
    p_quote_buf = ctypes.POINTER(ctypes.c_uint8)()
    quote_size = ctypes.c_uint32()
    
    # SHA512 = 64 bytes
    if len(payload_hash) != TDX_REPORT_DATA_SIZE:
        print("Error: Payload must be exactly 64 bytes")
        sys.exit(1)

    user_data = TdxReportData()
    user_data.d[:] = payload_hash

    # For testing only
    # user_data.d[:] = gen_report_data()

    # print_hex_dump("TDX report data: ", user_data.d)

    # Generate TD Report
    tdx_report = TdxReport()
    if tdx_attest.tdx_att_get_report(ctypes.byref(user_data), ctypes.byref(tdx_report)) != 0:
        print("Failed to get the TD report")
        return

    # Save report to file
    with open("user_report_data.dat", "wb") as f:
        f.write(bytearray(tdx_report.d))
    print("Wrote TD user report data to user_report_data.dat")

    # Generate TD Quote
    res = tdx_attest.tdx_att_get_quote(
        ctypes.byref(user_data), None, 0,
        ctypes.byref(selected_att_key_id),
        ctypes.byref(p_quote_buf), ctypes.byref(quote_size), 0
    )

    if res != 0:
        print("Failed to get the TD Quote")
        return
    else:
        print("Successfully get the TD Quote")

    # Convert pointer to bytes and save quote
    quote_bytes = ctypes.string_at(p_quote_buf, quote_size.value)
    # print_hex_dump("TDX quote: ", quote_bytes)

    with open(f"{output_dir}/quote.dat", "wb") as f:
        f.write(quote_bytes)
    print(f"Wrote TD Quote to {output_dir}/quote.dat")

    # Free the quote buffer
    tdx_attest.tdx_att_free_quote(p_quote_buf)

    return quote_bytes