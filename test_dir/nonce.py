def bits_to_target(bits):

import requests
import hashlib
import struct
import time

def bits_to_target(bits):
    exponent = bits >> 24
    mantissa = bits & 0xffffff
    return mantissa * (1 << (8 * (exponent - 3)))


def double_sha256(b):
    return hashlib.sha256(hashlib.sha256(b).digest()).digest()

def mine_block(version, prev_block, merkle_root, timestamp, bits, max_nonce=100000):
    target = bits_to_target(bits)
    print(f"Target: {hex(target)}")
    for nonce in range(max_nonce):
        # 构造真实区块头（80字节）
        header = (
            struct.pack('<L', version) +
            bytes.fromhex(prev_block)[::-1] +
            bytes.fromhex(merkle_root)[::-1] +
            struct.pack('<L', timestamp) +
            struct.pack('<L', bits) +
            struct.pack('<L', nonce)
        )
        hash_result = double_sha256(header)[::-1].hex()
        if int(hash_result, 16) < target:
            print(f"Success! Nonce: {nonce}, Hash: {hash_result}")
            return nonce, hash_result
        if nonce % 10000 == 0:
            print(f"Tried nonce: {nonce}, hash: {hash_result}")
    print("未找到合适的nonce")
    return None, None

try:
    # 获取最新区块hash
    latest_block_resp = requests.get("https://blockchain.info/latestblock", timeout=10)
    latest_block_resp.raise_for_status()
    block_hash = latest_block_resp.json()["hash"]

    # 获取完整区块数据
    block_resp = requests.get(f"https://blockchain.info/rawblock/{block_hash}", timeout=10)
    block_resp.raise_for_status()
    block = block_resp.json()

    version = block["ver"]
    prev_block = block["prev_block"]
    merkle_root = block["mrkl_root"]
    timestamp = block["time"]
    bits = block["bits"]

    print(f"区块头字段:")
    print(f"version: {version}")
    print(f"prev_block: {prev_block}")
    print(f"merkle_root: {merkle_root}")
    print(f"timestamp: {timestamp}")
    print(f"bits: {bits}")
    print("开始模拟真实区块头挖矿...")
    mine_block(version, prev_block, merkle_root, timestamp, bits)

except requests.exceptions.RequestException as e:
    print(f"Error fetching data: {e}")