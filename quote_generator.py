import base64
import os
import json 
import time 
import io
import hashlib

import torch
import tdx_quote_generator
import gpu_attestation

# For Gramine-SGX
def generate_quote(user_data) -> bytes:
    with open('/dev/attestation/user_report_data', 'wb') as f:
        f.write(user_data)
    with open('/dev/attestation/quote', 'rb') as f:
        quote = f.read()
    return quote

# For TDX TD quote generation with DCAP
def attest(payload, output_filename="./output_attestation.json", nonce: bytes = b""):
    json_data = {}
    os.makedirs(os.path.dirname(output_filename) or ".", exist_ok=True)

    # GPU attestation
    if torch.cuda.is_available():
        gpu_attestation_start = time.time()
        gpu_attestation_result = gpu_attestation.attest()
        json_data['gpu-attest'] = gpu_attestation_result
        payload['gpu-attest'] = gpu_attestation_result
        gpu_attestation_end = time.time()
        gpu_attestation_time = gpu_attestation_end - gpu_attestation_start

    payload_bytes = json.dumps(payload).encode('utf-8')

    output_measurement_start = time.time()
    hasher = hashlib.sha512()
    hasher.update(payload_bytes + nonce)
    payload_hash = hasher.digest()
    output_measurement_end = time.time()
    output_measurement_time = output_measurement_end - output_measurement_start

    # TD quote generation/attestation time
    attestion_start = time.time()
    quote = tdx_quote_generator.generate_quote(payload_hash, output_dir=os.path.dirname(output_filename))
    json_data['tdx-quote'] = base64.b64encode(quote).decode('utf-8')
    json_data['payload'] = base64.b64encode(payload_bytes).decode('utf-8')
    attestation_end = time.time()
    attestation_time = attestation_end - attestion_start
    if torch.cuda.is_available():
        attestation_time += gpu_attestation_time
    print("Time to form quote:", attestation_time, flush=True)

    output_storage_start = time.time()
    with open(output_filename, 'w') as f:
        json.dump(json_data, f, indent=2, default=str)
    output_storage_end = time.time()
    output_storage_time = output_storage_end - output_storage_start

    return output_measurement_time, attestation_time, output_storage_time