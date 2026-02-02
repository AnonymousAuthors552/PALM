from nv_attestation_sdk import attestation
import os
import json

NRAS_URL = "https://nras.attestation.nvidia.com/v3/attest/gpu"

def attest():
    client = attestation.Attestation()
    client.set_name("TDX_Guest_Platsec2")
    
    # Only for testing
    client.set_nonce("931d8dd0add203ac3d8b4fbde75e115278eefcdceac5b87671a748f32364dfcb")

    print("Attesting GPU...")
    print("[RemoteGPUTest] node name :", client.get_name())

    client.add_verifier(attestation.Devices.GPU, attestation.Environment.REMOTE, NRAS_URL, "")
    print(client.get_verifiers())

    print("[RemoteGPUTest] call get_evidence()")
    evidence_list = client.get_evidence()
    # print(evidence_list)

    print("[RemoteGPUTest] call attest() - expecting True")
    print(client.attest(evidence_list))

    # print("[RemoteGPUTest] token : " + str(client.get_token()))
    print("[RemoteGPUTest] call validate_token() - expecting True")

    file = "/shared/nvtrust/guest_tools/attestation_sdk/tests/policies/remote/v3/NVGPURemotePolicyExample.json"
    with open(os.path.join(os.path.dirname(__file__), file)) as json_file:
        json_data = json.load(json_file)
        remote_att_result_policy = json.dumps(json_data)

    res = client.validate_token(remote_att_result_policy)
    if res:
        print("Successfully attested the GPU")
    else:
        print("The policy does not match")

    return client.get_token()